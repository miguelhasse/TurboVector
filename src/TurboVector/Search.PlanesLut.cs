using System.Buffers;
using System.Buffers.Binary;
using System.Numerics;

namespace TurboVector;

public static partial class Search
{
    private sealed class LowPlaneQuery(byte[] masks, float unit, int sum, int rowLength)
    {
        public byte[] Masks { get; } = masks;
        public float Unit { get; } = unit;
        public int Sum { get; } = sum;
        public int RowLength { get; } = rowLength;
    }

    private static (float Scale, float Bias) BuildSignLut(
        ReadOnlySpan<float> query,
        float magnitude,
        byte cap,
        Span<byte> lut)
    {
        var groups = query.Length / 8;
        var products = ArrayPool<float>.Shared.Rent(groups * 32);
        var minima = ArrayPool<float>.Shared.Rent(groups * 2);
        Span<float> first = stackalloc float[4];
        Span<float> second = stackalloc float[4];
        var maxSpan = 0f;
        var bias = 0f;
        try
        {
            for (var group = 0; group < groups; group++)
            {
                for (var half = 0; half < 2; half++)
                {
                    var d = group * 8 + half * 4;
                    var p0 = query[d] * magnitude;
                    var p1 = query[d + 1] * magnitude;
                    var p2 = query[d + 2] * magnitude;
                    var p3 = query[d + 3] * magnitude;
                    first[0] = -p0 - p1;
                    first[1] = p1 - p0;
                    first[2] = p0 - p1;
                    first[3] = p0 + p1;
                    second[0] = -p2 - p3;
                    second[1] = p3 - p2;
                    second[2] = p2 - p3;
                    second[3] = p2 + p3;
                    var offset = group * 32 + half * 16;
                    for (var code = 0; code < 16; code++)
                    {
                        products[offset + code] = first[code >> 2] + second[code & 3];
                    }

                    var m01 = MathF.Abs(p0) + MathF.Abs(p1);
                    var m23 = MathF.Abs(p2) + MathF.Abs(p3);
                    var min = -m01 - m23;
                    var max = m01 + m23;
                    minima[group * 2 + half] = min;
                    bias += min;
                    if (max - min > maxSpan) maxSpan = max - min;
                }
            }

            var scale = maxSpan > 0f ? maxSpan / cap : 1f;
            if (scale < 1.17549435e-38f) scale = 1f;
            var inverseScale = 1f / scale;
            for (var table = 0; table < groups * 2; table++)
            {
                for (var code = 0; code < 16; code++)
                {
                    var offset = table * 16 + code;
                    lut[offset] = QuantizeToByte(products[offset], minima[table], inverseScale, cap);
                }
            }
            return (scale, bias);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(products);
            ArrayPool<float>.Shared.Return(minima);
        }
    }

    private static LowPlaneQuery BuildLowPlaneQuery(ReadOnlySpan<float> query, float magnitude)
    {
        var rowLength = query.Length / 8;
        var masks = new byte[checked(rowLength * 12)];
        var maximum = 0f;
        foreach (var q in query) maximum = MathF.Max(maximum, MathF.Abs(q));
        var inverse = maximum > 0f && float.IsFinite(maximum) ? 63f / maximum : 0f;
        var sum = 0;
        for (var d = 0; d < query.Length; d++)
        {
            var weight = Math.Min((uint)(MathF.Abs(query[d]) * inverse + 0.5f), 63u);
            if (weight == 0) continue;
            var negative = query[d] < 0f;
            sum += negative ? -(int)weight : (int)weight;
            var byteIndex = d / 8;
            var bit = (byte)(0x80 >> (d % 8));
            for (var plane = 0; plane < 6; plane++)
            {
                if (((weight >> plane) & 1) != 0)
                {
                    masks[((negative ? 6 : 0) + plane) * rowLength + byteIndex] |= bit;
                }
            }
        }

        return new LowPlaneQuery(masks, inverse > 0f ? magnitude / inverse : 0f, sum, rowLength);
    }

    private static float LowPlaneTerm(LowPlaneQuery query, ReadOnlySpan<byte> row)
    {
        var dot = 0L;
        for (var bit = 0; bit < 6; bit++)
        {
            var positive = bit * query.RowLength;
            var negative = (6 + bit) * query.RowLength;
            var count = 0;
            var i = 0;
            for (; i + sizeof(ulong) <= row.Length; i += sizeof(ulong))
            {
                var word = BinaryPrimitives.ReadUInt64LittleEndian(row.Slice(i, sizeof(ulong)));
                var pos = BinaryPrimitives.ReadUInt64LittleEndian(query.Masks.AsSpan(positive + i, sizeof(ulong)));
                var neg = BinaryPrimitives.ReadUInt64LittleEndian(query.Masks.AsSpan(negative + i, sizeof(ulong)));
                count += BitOperations.PopCount(word & pos) - BitOperations.PopCount(word & neg);
            }
            for (; i < row.Length; i++)
            {
                count += BitOperations.PopCount((uint)(row[i] & query.Masks[positive + i]));
                count -= BitOperations.PopCount((uint)(row[i] & query.Masks[negative + i]));
            }
            dot += (long)count << bit;
        }
        return query.Unit * (float)(2 * dot - query.Sum);
    }

    private static float TablePlaneTerm(
        ReadOnlySpan<byte> lut,
        float scale,
        float bias,
        ReadOnlySpan<byte> row)
    {
        var sum = 0u;
        for (var group = 0; group < row.Length; group++)
        {
            var code = row[group];
            sum += (uint)(lut[group * 32 + (code >> 4)] + lut[group * 32 + 16 + (code & 15)]);
        }
        return scale * sum + bias;
    }
}
