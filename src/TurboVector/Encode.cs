using System.Buffers;
using System.Numerics.Tensors;

namespace TurboVector;

/// <summary>
/// Encodes vectors by normalizing, rotating, quantizing, computing per-vector
/// correction scales, and bit-packing the quantized codes.
/// </summary>
public static class Encoder
{
    /// <summary>
    /// Encode vectors using the deterministic block-Hadamard rotation.
    /// </summary>
    public static (byte[] PackedCodes, float[] Scales) Encode(
        ReadOnlySpan<float> vectors,
        int n,
        int dim,
        Rotation rotation,
        ReadOnlySpan<float> boundaries,
        ReadOnlySpan<float> centroids,
        int bitWidth)
    {
        int packedLength = GetPackedLength(n, dim, bitWidth);
        byte[] packed = new byte[packedLength];
        float[] scales = new float[n];
        EncodeInto(vectors, n, dim, rotation, boundaries, centroids, bitWidth, packed, scales);
        return (packed, scales);
    }

    /// <summary>
    /// Encode a batch of vectors using the provided rotation and scalar quantizer.
    /// </summary>
    public static (byte[] PackedCodes, float[] Scales) Encode(
        ReadOnlySpan<float> vectors,
        int n,
        int dim,
        ReadOnlySpan<float> rotation,
        ReadOnlySpan<float> boundaries,
        ReadOnlySpan<float> centroids,
        int bitWidth)
    {
        int packedLength = GetPackedLength(n, dim, bitWidth);
        byte[] packed = new byte[packedLength];
        float[] scales = new float[n];
        EncodeInto(vectors, n, dim, rotation, boundaries, centroids, bitWidth, packed, scales);
        return (packed, scales);
    }

    /// <summary>
    /// Encode vectors directly into pre-grown destination spans, avoiding intermediate heap allocations.
    /// Caller must ensure packedDest.Length &gt;= n * (dim/8) * bitWidth and scalesDest.Length &gt;= n.
    /// </summary>
    public static void EncodeInto(
        ReadOnlySpan<float> vectors,
        int n,
        int dim,
        ReadOnlySpan<float> rotation,
        ReadOnlySpan<float> boundaries,
        ReadOnlySpan<float> centroids,
        int bitWidth,
        Span<byte> packedDest,
        Span<float> scalesDest)
    {
        ValidateInputs(vectors, n, dim, rotation, boundaries, centroids, bitWidth);

        int packedLength = GetPackedLength(n, dim, bitWidth);
        if (packedDest.Length < packedLength)
        {
            throw new ArgumentException($"packedDest length must be at least {packedLength}.", nameof(packedDest));
        }

        if (scalesDest.Length < n)
        {
            throw new ArgumentException($"scalesDest length must be at least {n}.", nameof(scalesDest));
        }

        float[]? norms = null;
        float[]? unit = null;
        float[]? rotated = null;
        byte[]? codes = null;
        try
        {
            norms = ArrayPool<float>.Shared.Rent(n);
            unit = ArrayPool<float>.Shared.Rent(checked(n * dim));
            rotated = ArrayPool<float>.Shared.Rent(checked(n * dim));
            codes = ArrayPool<byte>.Shared.Rent(checked(n * dim));

            for (int i = 0; i < n; i++)
            {
                int rowStart = i * dim;
                var vecRow = vectors.Slice(rowStart, dim);
                float norm = TensorPrimitives.Norm(vecRow);
                norms[i] = norm;
                float invNorm = norm > 1e-10f ? 1.0f / norm : 0.0f;
                TensorPrimitives.Multiply(vecRow, invNorm, unit.AsSpan(rowStart, dim));
            }

            for (int k = 0; k < dim; k++)
            {
                var rotRow = rotation.Slice(k * dim, dim);
                for (int i = 0; i < n; i++)
                {
                    int rowStart = i * dim;
                    rotated[rowStart + k] = TensorPrimitives.Dot(unit.AsSpan(rowStart, dim), rotRow);
                }
            }

            int rotatedLen = n * dim;
            for (int idx = 0; idx < rotatedLen; idx++)
            {
                float val = rotated[idx];
                byte code = 0;
                for (int b = 0; b < boundaries.Length; b++)
                {
                    code += val > boundaries[b] ? (byte)1 : (byte)0;
                }

                codes[idx] = code;
            }

            for (int i = 0; i < n; i++)
            {
                int rowStart = i * dim;
                float inner = 0.0f;
                for (int j = 0; j < dim; j++)
                {
                    inner += rotated[rowStart + j] * centroids[codes[rowStart + j]];
                }

                float clampedInner = Math.Max(inner, 1e-10f);
                scalesDest[i] = norms[i] / clampedInner;
            }

            PackCodesInto(codes.AsSpan(0, rotatedLen), n, dim, bitWidth, packedDest[..packedLength]);
        }
        finally
        {
            if (codes is not null) ArrayPool<byte>.Shared.Return(codes);
            if (rotated is not null) ArrayPool<float>.Shared.Return(rotated);
            if (unit is not null) ArrayPool<float>.Shared.Return(unit);
            if (norms is not null) ArrayPool<float>.Shared.Return(norms);
        }
    }

    /// <summary>
    /// Encode vectors directly with the block-Hadamard rotation, avoiding
    /// a dense dim-by-dim matrix and its quadratic construction cost.
    /// </summary>
    public static void EncodeInto(
        ReadOnlySpan<float> vectors,
        int n,
        int dim,
        Rotation rotation,
        ReadOnlySpan<float> boundaries,
        ReadOnlySpan<float> centroids,
        int bitWidth,
        Span<byte> packedDest,
        Span<float> scalesDest)
        => EncodeIntoCalibrated(
            vectors, n, dim, rotation, boundaries, centroids,
            ReadOnlySpan<float>.Empty, ReadOnlySpan<float>.Empty,
            bitWidth, packedDest, scalesDest);

    internal static void EncodeIntoCalibrated(
        ReadOnlySpan<float> vectors,
        int n,
        int dim,
        Rotation rotation,
        ReadOnlySpan<float> boundaries,
        ReadOnlySpan<float> centroids,
        ReadOnlySpan<float> tqplusShift,
        ReadOnlySpan<float> tqplusScale,
        int bitWidth,
        Span<byte> packedDest,
        Span<float> scalesDest)
    {
        ArgumentNullException.ThrowIfNull(rotation);
        ValidateRotationInputs(vectors, n, dim, rotation, boundaries, centroids, bitWidth);
        ValidateCalibration(dim, tqplusShift, tqplusScale);

        int packedLength = GetPackedLength(n, dim, bitWidth);
        if (packedDest.Length < packedLength)
        {
            throw new ArgumentException($"packedDest length must be at least {packedLength}.", nameof(packedDest));
        }

        if (scalesDest.Length < n)
        {
            throw new ArgumentException($"scalesDest length must be at least {n}.", nameof(scalesDest));
        }

        float[]? rotated = null;
        float[]? scratch = null;
        try
        {
            int vectorLength = checked(n * dim);
            rotated = ArrayPool<float>.Shared.Rent(vectorLength);
            scratch = ArrayPool<float>.Shared.Rent(dim);

            for (int i = 0; i < n; i++)
            {
                int rowStart = i * dim;
                ReadOnlySpan<float> source = vectors.Slice(rowStart, dim);
                float norm = ComputeNorm(source);
                scalesDest[i] = norm;
                float inverseNorm = norm > 1e-10f ? 1.0f / norm : 0.0f;
                rotation.ApplyScaledInto(source, inverseNorm, rotated.AsSpan(rowStart, dim), scratch.AsSpan(0, dim));
            }

            EncodePrerotatedInto(
                rotated.AsSpan(0, vectorLength),
                scalesDest[..n],
                n,
                dim,
                boundaries,
                centroids,
                tqplusShift,
                tqplusScale,
                bitWidth,
                packedDest[..packedLength],
                scalesDest);
        }
        finally
        {
            if (scratch is not null) ArrayPool<float>.Shared.Return(scratch);
            if (rotated is not null) ArrayPool<float>.Shared.Return(rotated);
        }
    }

    internal static void EncodePrerotatedInto(
        ReadOnlySpan<float> rotated,
        ReadOnlySpan<float> norms,
        int n,
        int dim,
        ReadOnlySpan<float> boundaries,
        ReadOnlySpan<float> centroids,
        ReadOnlySpan<float> tqplusShift,
        ReadOnlySpan<float> tqplusScale,
        int bitWidth,
        Span<byte> packedDest,
        Span<float> scalesDest)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(n);
        InputValidation.ValidateDimension(dim, nameof(dim));
        InputValidation.ValidateBitWidth(bitWidth, nameof(bitWidth));
        ValidateCalibration(dim, tqplusShift, tqplusScale);

        if (rotated.Length != checked(n * dim) || norms.Length != n)
        {
            throw new ArgumentException("rotated and norms lengths must equal n * dim and n.");
        }

        int packedLength = GetPackedLength(n, dim, bitWidth);
        if (packedDest.Length < packedLength || scalesDest.Length < n)
        {
            throw new ArgumentException("Destination spans are too short.");
        }

        int levels = 1 << bitWidth;
        if (boundaries.Length != levels - 1 || centroids.Length != levels)
        {
            throw new ArgumentException("Codebook lengths do not match bitWidth.");
        }

        float[]? inverseTqplusScale = null;
        byte[]? codes = null;
        try
        {
            if (!tqplusScale.IsEmpty)
            {
                inverseTqplusScale = ArrayPool<float>.Shared.Rent(dim);
                for (int d = 0; d < dim; d++)
                {
                    inverseTqplusScale[d] = 1.0f / tqplusScale[d];
                }
            }

            int vectorLength = checked(n * dim);
            codes = ArrayPool<byte>.Shared.Rent(vectorLength);
            for (int row = 0; row < n; row++)
            {
                int rowStart = row * dim;
                double chain0 = 0.0;
                double chain1 = 0.0;
                double chain2 = 0.0;
                double chain3 = 0.0;

                for (int d = 0; d < dim; d++)
                {
                    float value = tqplusShift.IsEmpty
                        ? rotated[rowStart + d]
                        : (rotated[rowStart + d] + tqplusShift[d]) * tqplusScale[d];
                    byte code = 0;
                    for (int boundary = 0; boundary < boundaries.Length; boundary++)
                    {
                        if (value > boundaries[boundary])
                        {
                            code++;
                        }
                    }

                    codes[rowStart + d] = code;
                    double reconstructed = tqplusShift.IsEmpty
                        ? centroids[code]
                        : ((double)centroids[code] * inverseTqplusScale![d]) - tqplusShift[d];
                    double term = (double)rotated[rowStart + d] * reconstructed;
                    switch (d & 3)
                    {
                        case 0: chain0 += term; break;
                        case 1: chain1 += term; break;
                        case 2: chain2 += term; break;
                        default: chain3 += term; break;
                    }
                }

                double inner = (chain0 + chain1) + (chain2 + chain3);
                scalesDest[row] = inner > 0.1 ? norms[row] / (float)inner : 0.0f;
            }

            PackCodesInto(codes.AsSpan(0, vectorLength), n, dim, bitWidth, packedDest[..packedLength]);
        }
        finally
        {
            if (codes is not null) ArrayPool<byte>.Shared.Return(codes);
            if (inverseTqplusScale is not null) ArrayPool<float>.Shared.Return(inverseTqplusScale);
        }
    }

    private static void ValidateInputs(
        ReadOnlySpan<float> vectors,
        int n,
        int dim,
        ReadOnlySpan<float> rotation,
        ReadOnlySpan<float> boundaries,
        ReadOnlySpan<float> centroids,
        int bitWidth)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(n);
        InputValidation.ValidateDimension(dim, nameof(dim));
        InputValidation.ValidateBitWidth(bitWidth, nameof(bitWidth));

        if (vectors.Length != checked(n * dim))
        {
            throw new ArgumentException("vectors length must equal n * dim.", nameof(vectors));
        }

        InputValidation.ValidateValues(vectors, dim, nameof(vectors));

        if (rotation.Length != checked(dim * dim))
        {
            throw new ArgumentException("rotation length must equal dim * dim.", nameof(rotation));
        }

        int levels = 1 << bitWidth;
        if (boundaries.Length != levels - 1)
        {
            throw new ArgumentException("boundaries length must equal (1 << bitWidth) - 1.", nameof(boundaries));
        }

        if (centroids.Length != levels)
        {
            throw new ArgumentException("centroids length must equal 1 << bitWidth.", nameof(centroids));
        }

        if (!IsSorted(boundaries))
        {
            throw new InvalidOperationException("Codebook boundaries must be monotonically increasing.");
        }
    }

    private static void ValidateRotationInputs(
        ReadOnlySpan<float> vectors,
        int n,
        int dim,
        Rotation rotation,
        ReadOnlySpan<float> boundaries,
        ReadOnlySpan<float> centroids,
        int bitWidth)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(n);
        InputValidation.ValidateDimension(dim, nameof(dim));
        InputValidation.ValidateBitWidth(bitWidth, nameof(bitWidth));
        if (rotation.Dim != dim)
        {
            throw new ArgumentException("rotation dimension must equal dim.", nameof(rotation));
        }

        if (vectors.Length != checked(n * dim))
        {
            throw new ArgumentException("vectors length must equal n * dim.", nameof(vectors));
        }

        InputValidation.ValidateValues(vectors, dim, nameof(vectors));
        int levels = 1 << bitWidth;
        if (boundaries.Length != levels - 1)
        {
            throw new ArgumentException("boundaries length must equal (1 << bitWidth) - 1.", nameof(boundaries));
        }

        if (centroids.Length != levels)
        {
            throw new ArgumentException("centroids length must equal 1 << bitWidth.", nameof(centroids));
        }

        if (!IsSorted(boundaries))
        {
            throw new InvalidOperationException("Codebook boundaries must be monotonically increasing.");
        }
    }

    internal static float ComputeNorm(ReadOnlySpan<float> values)
    {
        Span<float> chains = stackalloc float[8];
        chains.Clear();
        for (int i = 0; i < values.Length; i++)
        {
            float value = values[i];
            chains[i & 7] += value * value;
        }

        float sum = ((chains[0] + chains[1]) + (chains[2] + chains[3]))
            + ((chains[4] + chains[5]) + (chains[6] + chains[7]));
        return MathF.Sqrt(sum);
    }

    private static void ValidateCalibration(int dim, ReadOnlySpan<float> shift, ReadOnlySpan<float> scale)
    {
        if (shift.IsEmpty && scale.IsEmpty)
        {
            return;
        }

        if (shift.Length != dim || scale.Length != dim)
        {
            throw new ArgumentException("Calibration shift and scale must both be empty or have length dim.");
        }
    }

    private static int GetPackedLength(int n, int dim, int bits)
    {
        return checked(n * (dim / 8) * bits);
    }

    private static void PackCodesInto(ReadOnlySpan<byte> codes, int n, int dim, int bits, Span<byte> dest)
    {
        int bytesPerPlane = dim / 8;
        int bytesPerRow = bits * bytesPerPlane;
        dest.Clear();
        for (int i = 0; i < n; i++)
        {
            int rowStart = i * dim;
            int packedRowStart = i * bytesPerRow;
            for (int p = 0; p < bits; p++)
            {
                int planeStart = packedRowStart + (p * bytesPerPlane);
                for (int j = 0; j < dim; j++)
                {
                    byte code = codes[rowStart + j];
                    if ((code & (1 << p)) != 0)
                    {
                        int bytePos = j >> 3;
                        int bitPos = 7 - (j & 7);
                        dest[planeStart + bytePos] |= (byte)(1 << bitPos);
                    }
                }
            }
        }
    }

    private static byte[] PackCodes(ReadOnlySpan<byte> codes, int n, int dim, int bits)
    {
        byte[] packed = new byte[GetPackedLength(n, dim, bits)];
        PackCodesInto(codes, n, dim, bits, packed);
        return packed;
    }

    private static bool IsSorted(ReadOnlySpan<float> values)
    {
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] < values[i - 1])
            {
                return false;
            }
        }

        return true;
    }
}
