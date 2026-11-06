using System.Buffers.Binary;
using TurboVector;

internal static class EncodeHashExample
{
    private static readonly (int Dim, int Bits)[] Cells =
    [
        (200, 2),
        (600, 3),
        (768, 4),
        (1000, 4),
        (1024, 3),
        (1536, 2),
        (1536, 4),
        (3072, 4),
    ];

    public static void Run(string[] args)
    {
        if (args.Length != 0)
        {
            throw new ArgumentException("encode-hash takes no options.");
        }

        Console.Error.WriteLine(
            $"# arch={System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture} " +
            $"os={Environment.OSVersion.Platform} cells={Cells.Length}");

        foreach ((int dim, int bits) in Cells)
        {
            const int rows = 2000;
            ulong seed = 0x7B0_0000UL ^ ((ulong)dim << 8) ^ (uint)bits;
            float[] vectors = CreateVectors(rows, dim, seed);
            var index = new TurboQuantIndex(dim, bits);
            index.Calibrate(vectors);
            index.Add(vectors);

            var (boundaries, centroids) = Codebook.Compute(bits, dim);
            byte[] image = index.ToBytes();
            Console.WriteLine(
                $"dim={dim} bits={bits} " +
                $"boundaries={Hash(boundaries):x16} centroids={Hash(centroids):x16} " +
                $"calibration={HashCalibration(index.TqPlusShift, index.TqPlusScale):x16} " +
                $"codes={Hash(index.PackedCodes):x16} scales={Hash(index.Scales):x16} " +
                $"file={Hash(image):x16}");
        }
    }

    private static float[] CreateVectors(int rows, int dim, ulong seed)
    {
        float[] vectors = new float[checked(rows * dim)];
        ulong state = seed | 1UL;
        for (int i = 0; i < vectors.Length; i++)
        {
            state = unchecked(state * 6_364_136_223_846_793_005UL + 1_442_695_040_888_963_407UL);
            uint high = (uint)(state >> 32);
            vectors[i] = (float)(high / 2_147_483_648.0 - 1.0);
        }

        return vectors;
    }

    private static ulong HashCalibration(ReadOnlySpan<float> shift, ReadOnlySpan<float> scale)
    {
        ulong hash = HashStart;
        hash = Append(hash, shift);
        return Append(hash, scale);
    }

    private static ulong Hash(ReadOnlySpan<float> values) => Append(HashStart, values);

    private static ulong Append(ulong hash, ReadOnlySpan<float> values)
    {
        Span<byte> bytes = stackalloc byte[sizeof(float)];
        foreach (float value in values)
        {
            BinaryPrimitives.WriteInt32LittleEndian(bytes, BitConverter.SingleToInt32Bits(value));
            hash = AddByte(hash, bytes[0]);
            hash = AddByte(hash, bytes[1]);
            hash = AddByte(hash, bytes[2]);
            hash = AddByte(hash, bytes[3]);
        }

        return hash;
    }

    private static ulong Hash(ReadOnlySpan<byte> values)
    {
        ulong hash = HashStart;
        foreach (byte value in values)
        {
            hash = AddByte(hash, value);
        }

        return hash;
    }

    private const ulong HashStart = 0xcbf29ce484222325;

    private static ulong AddByte(ulong hash, byte value)
        => unchecked((hash ^ value) * 0x100000001b3);
}
