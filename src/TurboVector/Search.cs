using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Threading;
using System.Threading.Tasks;

namespace TurboVector;

/// <summary>
/// Quantized similarity search over blocked codebooks.
/// </summary>
public static class Search
{
    private static long _blocksSkippedByMask;

    /// <summary>
    /// Total number of 32-vector blocks skipped due to mask filtering, accumulated
    /// across ALL index instances and ALL queries in every search since the last
    /// <see cref="ResetBlocksSkippedByMask"/> call. Call <see cref="ResetBlocksSkippedByMask"/>
    /// before a search and read the delta afterwards to get per-search metrics.
    /// </summary>
    public static long BlocksSkippedByMask => Interlocked.Read(ref _blocksSkippedByMask);

    public static void ResetBlocksSkippedByMask()
    {
        Interlocked.Exchange(ref _blocksSkippedByMask, 0L);
    }

    public static (float[] Scores, long[] Indices) RunSearch(
        ReadOnlySpan<float> queries,
        int nq,
        ReadOnlySpan<float> rotation,
        ReadOnlySpan<byte> packedCodes,
        ReadOnlySpan<float> centroids,
        ReadOnlySpan<float> vecScales,
        int bitWidth,
        int dim,
        int nVectors,
        int k,
        ReadOnlySpan<ulong> mask)
    {
        ValidateInputs(queries, nq, rotation, centroids, vecScales, bitWidth, dim, nVectors, k, mask);

        return bitWidth switch
        {
            2 or 4 => RunPackedNibbleSearch(queries, nq, rotation, packedCodes, centroids, vecScales, bitWidth, dim, nVectors, k, mask),
            3 => RunPacked3BitSearch(queries, nq, rotation, packedCodes, centroids, vecScales, dim, nVectors, k, mask),
            _ => throw new NotSupportedException("Only 2-bit, 3-bit, and 4-bit search are supported."),
        };
    }

    public static (float[] Scores, long[] Indices) Search_(
        ReadOnlySpan<float> queries,
        int nq,
        ReadOnlySpan<float> rotation,
        ReadOnlySpan<byte> blockedCodes,
        ReadOnlySpan<float> centroids,
        ReadOnlySpan<float> vecScales,
        int bitWidth,
        int dim,
        int nVectors,
        int nBlocks,
        int k,
        ReadOnlySpan<ulong> mask)
    {
        ValidateInputs(queries, nq, rotation, centroids, vecScales, bitWidth, dim, nVectors, k, mask);

        if (bitWidth == 3)
        {
            throw new NotSupportedException("Use RunSearch with packed 3-bit codes or Search3Bit_ with blocked sub-codes and plane2.");
        }

        int nByteGroups = GetNByteGroups(bitWidth, dim);
        int expectedBlockedLength = checked(nBlocks * nByteGroups * Constants.Block);
        if (blockedCodes.Length != expectedBlockedLength)
        {
            throw new ArgumentException($"Expected blockedCodes length {expectedBlockedLength}, got {blockedCodes.Length}.", nameof(blockedCodes));
        }

        return RunBlockedNibbleSearch(
            queries,
            nq,
            rotation,
            blockedCodes.ToArray(),
            centroids.ToArray(),
            vecScales.ToArray(),
            bitWidth,
            dim,
            nVectors,
            nBlocks,
            k,
            mask.IsEmpty ? null : mask.ToArray(),
            mask.Length);
    }

    internal static (float[] Scores, long[] Indices) Search_(
        ReadOnlySpan<float> queries,
        int nq,
        Rotation rotation,
        byte[] blockedCodes,
        float[] centroids,
        float[] vecScales,
        int bitWidth,
        int dim,
        int nVectors,
        int nBlocks,
        int k,
        ulong[]? mask,
        int maskLength,
        float[] tqplusShift,
        float[] tqplusScale)
    {
        ArgumentNullException.ThrowIfNull(rotation);
        ArgumentNullException.ThrowIfNull(blockedCodes);
        ArgumentNullException.ThrowIfNull(centroids);
        ArgumentNullException.ThrowIfNull(vecScales);
        if (maskLength < 0 || (mask is null ? maskLength != 0 : maskLength > mask.Length))
        {
            throw new ArgumentOutOfRangeException(nameof(maskLength));
        }

        ReadOnlySpan<float> scales = vecScales.AsSpan(0, nVectors);
        ReadOnlySpan<ulong> maskSpan = mask is null ? ReadOnlySpan<ulong>.Empty : mask.AsSpan(0, maskLength);
        ValidateInputs(queries, nq, centroids, scales, bitWidth, dim, nVectors, k, maskSpan);
        if (rotation.Dim != dim)
        {
            throw new ArgumentException("rotation dimension must equal dim.", nameof(rotation));
        }
        ValidateCalibration(dim, tqplusShift, tqplusScale);

        if (bitWidth == 3)
        {
            throw new NotSupportedException("Use Search3Bit_ for blocked 3-bit codes.");
        }

        int nByteGroups = GetNByteGroups(bitWidth, dim);
        int expectedLength = checked(nBlocks * nByteGroups * Constants.Block);
        if (blockedCodes.Length != expectedLength)
        {
            throw new ArgumentException($"Expected blockedCodes length {expectedLength}, got {blockedCodes.Length}.", nameof(blockedCodes));
        }

        return RunBlockedNibbleSearch(
            queries, nq, default, blockedCodes, centroids, vecScales, bitWidth,
            dim, nVectors, nBlocks, k, mask, maskLength, rotation, tqplusShift, tqplusScale);
    }

    public static (float[] Scores, long[] Indices) Search_(
        ReadOnlySpan<float> queries,
        int nq,
        ReadOnlySpan<float> rotation,
        byte[] blockedCodes,
        float[] centroids,
        float[] vecScales,
        int bitWidth,
        int dim,
        int nVectors,
        int nBlocks,
        int k,
        ulong[]? mask,
        int maskLength,
        float[] tqplusShift,
        float[] tqplusScale)
    {
        ArgumentNullException.ThrowIfNull(blockedCodes);
        ArgumentNullException.ThrowIfNull(centroids);
        ArgumentNullException.ThrowIfNull(vecScales);
        if (maskLength < 0 || (mask is null ? maskLength != 0 : maskLength > mask.Length))
        {
            throw new ArgumentOutOfRangeException(nameof(maskLength));
        }

        ReadOnlySpan<float> vecScalesSpan = vecScales.AsSpan(0, nVectors);
        ReadOnlySpan<ulong> maskSpan = mask is null ? ReadOnlySpan<ulong>.Empty : mask.AsSpan(0, maskLength);
        ValidateInputs(queries, nq, rotation, centroids, vecScalesSpan, bitWidth, dim, nVectors, k, maskSpan);

        if (bitWidth == 3)
        {
            throw new NotSupportedException("Use RunSearch with packed 3-bit codes or Search3Bit_ with blocked sub-codes and plane2.");
        }

        int nByteGroups = GetNByteGroups(bitWidth, dim);
        int expectedBlockedLength = checked(nBlocks * nByteGroups * Constants.Block);
        if (blockedCodes.Length != expectedBlockedLength)
        {
            throw new ArgumentException($"Expected blockedCodes length {expectedBlockedLength}, got {blockedCodes.Length}.", nameof(blockedCodes));
        }

        return RunBlockedNibbleSearch(
            queries,
            nq,
            rotation,
            blockedCodes,
            centroids,
            vecScales,
            bitWidth,
            dim,
            nVectors,
            nBlocks,
            k,
            mask,
            maskLength);
    }

    public static (float[] Scores, long[] Indices) Search3Bit_(
        ReadOnlySpan<float> queries,
        int nq,
        ReadOnlySpan<float> rotation,
        ReadOnlySpan<byte> subCodes,
        ReadOnlySpan<byte> plane2Blocked,
        ReadOnlySpan<float> centroids,
        ReadOnlySpan<float> vecScales,
        int dim,
        int nVectors,
        int nBlocks,
        int k,
        ReadOnlySpan<ulong> mask)
    {
        ValidateInputs(queries, nq, rotation, centroids, vecScales, 3, dim, nVectors, k, mask);

        int subByteGroups = dim / 4;
        int plane2ByteGroups = dim / 8;
        int expectedSubCodesLength = checked(nBlocks * subByteGroups * Constants.Block);
        int expectedPlane2Length = checked(nBlocks * plane2ByteGroups * Constants.Block);
        if (subCodes.Length != expectedSubCodesLength)
        {
            throw new ArgumentException($"Expected subCodes length {expectedSubCodesLength}, got {subCodes.Length}.", nameof(subCodes));
        }

        if (plane2Blocked.Length != expectedPlane2Length)
        {
            throw new ArgumentException($"Expected plane2Blocked length {expectedPlane2Length}, got {plane2Blocked.Length}.", nameof(plane2Blocked));
        }

        return RunBlocked3BitSearch(
            queries,
            nq,
            rotation,
            subCodes.ToArray(),
            plane2Blocked.ToArray(),
            centroids.ToArray(),
            vecScales.ToArray(),
            dim,
            nVectors,
            nBlocks,
            k,
            mask.IsEmpty ? null : mask.ToArray(),
            mask.Length);
    }

    public static (float[] Scores, long[] Indices) Search3Bit_(
        ReadOnlySpan<float> queries,
        int nq,
        ReadOnlySpan<float> rotation,
        byte[] subCodes,
        byte[] plane2Blocked,
        float[] centroids,
        float[] vecScales,
        int dim,
        int nVectors,
        int nBlocks,
        int k,
        ulong[]? mask,
        int maskLength)
    {
        ArgumentNullException.ThrowIfNull(subCodes);
        ArgumentNullException.ThrowIfNull(plane2Blocked);
        ArgumentNullException.ThrowIfNull(centroids);
        ArgumentNullException.ThrowIfNull(vecScales);
        if (maskLength < 0 || (mask is null ? maskLength != 0 : maskLength > mask.Length))
        {
            throw new ArgumentOutOfRangeException(nameof(maskLength));
        }

        ReadOnlySpan<float> vecScalesSpan = vecScales.AsSpan(0, nVectors);
        ReadOnlySpan<ulong> maskSpan = mask is null ? ReadOnlySpan<ulong>.Empty : mask.AsSpan(0, maskLength);
        ValidateInputs(queries, nq, rotation, centroids, vecScalesSpan, 3, dim, nVectors, k, maskSpan);

        int subByteGroups = dim / 4;
        int plane2ByteGroups = dim / 8;
        int expectedSubCodesLength = checked(nBlocks * subByteGroups * Constants.Block);
        int expectedPlane2Length = checked(nBlocks * plane2ByteGroups * Constants.Block);
        if (subCodes.Length != expectedSubCodesLength)
        {
            throw new ArgumentException($"Expected subCodes length {expectedSubCodesLength}, got {subCodes.Length}.", nameof(subCodes));
        }

        if (plane2Blocked.Length != expectedPlane2Length)
        {
            throw new ArgumentException($"Expected plane2Blocked length {expectedPlane2Length}, got {plane2Blocked.Length}.", nameof(plane2Blocked));
        }

        return RunBlocked3BitSearch(
            queries,
            nq,
            rotation,
            subCodes,
            plane2Blocked,
            centroids,
            vecScales,
            dim,
            nVectors,
            nBlocks,
            k,
            mask,
            maskLength);
    }

    internal static (float[] Scores, long[] Indices) Search3Bit_(
        ReadOnlySpan<float> queries,
        int nq,
        Rotation rotation,
        byte[] subCodes,
        byte[] plane2Blocked,
        float[] centroids,
        float[] vecScales,
        int dim,
        int nVectors,
        int nBlocks,
        int k,
        ulong[]? mask,
        int maskLength,
        float[] tqplusShift,
        float[] tqplusScale)
    {
        ArgumentNullException.ThrowIfNull(rotation);
        ArgumentNullException.ThrowIfNull(subCodes);
        ArgumentNullException.ThrowIfNull(plane2Blocked);
        ArgumentNullException.ThrowIfNull(centroids);
        ArgumentNullException.ThrowIfNull(vecScales);
        if (maskLength < 0 || (mask is null ? maskLength != 0 : maskLength > mask.Length))
        {
            throw new ArgumentOutOfRangeException(nameof(maskLength));
        }

        ReadOnlySpan<float> scales = vecScales.AsSpan(0, nVectors);
        ReadOnlySpan<ulong> maskSpan = mask is null ? ReadOnlySpan<ulong>.Empty : mask.AsSpan(0, maskLength);
        ValidateInputs(queries, nq, centroids, scales, 3, dim, nVectors, k, maskSpan);
        if (rotation.Dim != dim)
        {
            throw new ArgumentException("rotation dimension must equal dim.", nameof(rotation));
        }
        ValidateCalibration(dim, tqplusShift, tqplusScale);

        int subLength = checked(nBlocks * (dim / 4) * Constants.Block);
        int planeLength = checked(nBlocks * (dim / 8) * Constants.Block);
        if (subCodes.Length != subLength)
        {
            throw new ArgumentException($"Expected subCodes length {subLength}, got {subCodes.Length}.", nameof(subCodes));
        }

        if (plane2Blocked.Length != planeLength)
        {
            throw new ArgumentException($"Expected plane2Blocked length {planeLength}, got {plane2Blocked.Length}.", nameof(plane2Blocked));
        }

        return RunBlocked3BitSearch(
            queries, nq, default, subCodes, plane2Blocked, centroids, vecScales,
            dim, nVectors, nBlocks, k, mask, maskLength, rotation, tqplusShift, tqplusScale);
    }

    private static (float[] Scores, long[] Indices) RunPackedNibbleSearch(
        ReadOnlySpan<float> queries,
        int nq,
        ReadOnlySpan<float> rotation,
        ReadOnlySpan<byte> packedCodes,
        ReadOnlySpan<float> centroids,
        ReadOnlySpan<float> vecScales,
        int bitWidth,
        int dim,
        int nVectors,
        int k,
        ReadOnlySpan<ulong> mask)
    {
        int bytesPerPlane = dim / 8;
        int expectedPackedLength = checked(nVectors * bytesPerPlane * bitWidth);
        if (packedCodes.Length != expectedPackedLength)
        {
            throw new ArgumentException($"Expected packedCodes length {expectedPackedLength}, got {packedCodes.Length}.", nameof(packedCodes));
        }

        var (blocked, nBlocks) = Pack.Repack(packedCodes, nVectors, bitWidth, dim);
        return RunBlockedNibbleSearch(
            queries,
            nq,
            rotation,
            blocked,
            centroids.ToArray(),
            vecScales.ToArray(),
            bitWidth,
            dim,
            nVectors,
            nBlocks,
            k,
            mask.IsEmpty ? null : mask.ToArray(),
            mask.Length);
    }

    private static (float[] Scores, long[] Indices) RunPacked3BitSearch(
        ReadOnlySpan<float> queries,
        int nq,
        ReadOnlySpan<float> rotation,
        ReadOnlySpan<byte> packedCodes,
        ReadOnlySpan<float> centroids,
        ReadOnlySpan<float> vecScales,
        int dim,
        int nVectors,
        int k,
        ReadOnlySpan<ulong> mask)
    {
        int bytesPerPlane = dim / 8;
        int expectedPackedLength = checked(nVectors * bytesPerPlane * 3);
        if (packedCodes.Length != expectedPackedLength)
        {
            throw new ArgumentException($"Expected packedCodes length {expectedPackedLength}, got {packedCodes.Length}.", nameof(packedCodes));
        }

        var (subCodes, plane2Blocked, nBlocks) = Pack.Repack3Bit(packedCodes, nVectors, dim);
        return RunBlocked3BitSearch(
            queries,
            nq,
            rotation,
            subCodes,
            plane2Blocked,
            centroids.ToArray(),
            vecScales.ToArray(),
            dim,
            nVectors,
            nBlocks,
            k,
            mask.IsEmpty ? null : mask.ToArray(),
            mask.Length);
    }

    private static (float[] Scores, long[] Indices) RunBlockedNibbleSearch(
        ReadOnlySpan<float> queries,
        int nq,
        ReadOnlySpan<float> rotation,
        byte[] blockedCodes,
        float[] centroids,
        float[] vecScales,
        int bitWidth,
        int dim,
        int nVectors,
        int nBlocks,
        int k,
        ulong[]? mask,
        int maskLength,
        Rotation? rotationPlan = null,
        float[]? tqplusShift = null,
        float[]? tqplusScale = null)
    {
        ReadOnlySpan<ulong> maskSpan = mask is null ? ReadOnlySpan<ulong>.Empty : mask.AsSpan(0, maskLength);

        // Compute effectiveK and allocate result arrays BEFORE the expensive rotation so
        // that we short-circuit early and also catch any overflow before doing work.
        int effectiveK = GetEffectiveK(nVectors, k, maskSpan);
        float[] allScores = new float[checked(nq * effectiveK)];
        long[] allIndices = new long[checked(nq * effectiveK)];

        if (effectiveK == 0)
        {
            return (allScores, allIndices);
        }

        int nByteGroups = GetNByteGroups(bitWidth, dim);
        ReadOnlySpan<byte> blockedSpan = blockedCodes.AsSpan(0, checked(nBlocks * nByteGroups * Constants.Block));
        ReadOnlySpan<float> vecScalesSpan = vecScales.AsSpan(0, nVectors);
        ReadOnlySpan<float> centroidsSpan = centroids;

        int rotSize = checked(nq * dim);
        float[]? rotBuf = ArrayPool<float>.Shared.Rent(rotSize);
        try
        {
            if (rotationPlan is null)
            {
                FillRotatedQueries(queries, nq, dim, rotation, rotBuf.AsSpan(0, rotSize));
            }
            else
            {
                FillRotatedQueries(queries, nq, dim, rotationPlan, rotBuf.AsSpan(0, rotSize));
            }
            float[]? biasCorrections = ApplyQueryCalibration(
                rotBuf, nq, dim, tqplusShift, tqplusScale);
            float[] rotatedQueries = rotBuf; // alias for clarity; safe as Parallel.For is synchronous

            // Collect results into per-query local arrays inside the parallel loop to avoid
            // false sharing on allScores/allIndices when effectiveK is small (< 16 floats
            // or < 8 longs fit in one 64-byte cache line, causing cross-thread contention).
            var perQueryScores = new float[nq][];
            var perQueryIndices = new int[nq][];
            int lutSize = bitWidth switch
            {
                2 => checked(nByteGroups * 16),
                4 => checked(nByteGroups * 32),
                _ => throw new NotSupportedException("Nibble search only supports 2-bit or 4-bit codes."),
            };

            // Avoid Parallel.For scheduling overhead for single-query searches.
            if (nq == 1)
            {
                byte[] lut = ArrayPool<byte>.Shared.Rent(lutSize);
                try
                {
                    var qRot = new ReadOnlySpan<float>(rotatedQueries, 0, dim);
                    var lutSpan = lut.AsSpan(0, lutSize);
                    var (scale, bias) = bitWidth switch
                    {
                        2 => BuildLut2BitInto(qRot, centroidsSpan, lutSpan),
                        4 => BuildLut4BitInto(qRot, centroidsSpan, lutSpan),
                        _ => throw new NotSupportedException("Nibble search only supports 2-bit or 4-bit codes."),
                    };
                    if (biasCorrections is not null)
                    {
                        bias += biasCorrections[0];
                    }

                    var (scores, indices) = bitWidth switch
                    {
                        2 => Score2Bit(blockedSpan, lut, scale, bias, vecScalesSpan, dim, nVectors, nBlocks, effectiveK, maskSpan),
                        4 => Score4Bit(blockedSpan, lut, scale, bias, vecScalesSpan, dim, nVectors, nBlocks, effectiveK, maskSpan),
                        _ => throw new NotSupportedException("Nibble search only supports 2-bit or 4-bit codes."),
                    };
                    perQueryScores[0] = scores;
                    perQueryIndices[0] = indices;
                }
                finally
                {
                    ArrayPool<byte>.Shared.Return(lut);
                }
            }
            else
            {
                Parallel.For(0, nq, qi =>
                {
                    byte[] lut = ArrayPool<byte>.Shared.Rent(lutSize);
                    try
                    {
                        var qRot = new ReadOnlySpan<float>(rotatedQueries, qi * dim, dim);
                        var lutSpan = lut.AsSpan(0, lutSize);
                        var qMask = mask is null ? ReadOnlySpan<ulong>.Empty : mask.AsSpan(0, maskLength);
                        var (scale, bias) = bitWidth switch
                        {
                            2 => BuildLut2BitInto(qRot, centroids, lutSpan),
                            4 => BuildLut4BitInto(qRot, centroids, lutSpan),
                            _ => throw new NotSupportedException("Nibble search only supports 2-bit or 4-bit codes."),
                        };
                        if (biasCorrections is not null)
                        {
                            bias += biasCorrections[qi];
                        }

                        var (scores, indices) = bitWidth switch
                        {
                            2 => Score2Bit(blockedCodes, lut, scale, bias, vecScales, dim, nVectors, nBlocks, effectiveK, qMask),
                            4 => Score4Bit(blockedCodes, lut, scale, bias, vecScales, dim, nVectors, nBlocks, effectiveK, qMask),
                            _ => throw new NotSupportedException("Nibble search only supports 2-bit or 4-bit codes."),
                        };

                        perQueryScores[qi] = scores;
                        perQueryIndices[qi] = indices;
                    }
                    finally
                    {
                        ArrayPool<byte>.Shared.Return(lut);
                    }
                });
            }

            for (int qi = 0; qi < nq; qi++)
            {
                Array.Copy(perQueryScores[qi], 0, allScores, qi * effectiveK, effectiveK);
                int[] qIdx = perQueryIndices[qi];
                for (int i = 0; i < effectiveK; i++)
                {
                    allIndices[(qi * effectiveK) + i] = qIdx[i];
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rotBuf);
        }

        return (allScores, allIndices);
    }

    private static (float[] Scores, long[] Indices) RunBlocked3BitSearch(
        ReadOnlySpan<float> queries,
        int nq,
        ReadOnlySpan<float> rotation,
        byte[] subCodes,
        byte[] plane2Blocked,
        float[] centroids,
        float[] vecScales,
        int dim,
        int nVectors,
        int nBlocks,
        int k,
        ulong[]? mask,
        int maskLength,
        Rotation? rotationPlan = null,
        float[]? tqplusShift = null,
        float[]? tqplusScale = null)
    {
        ReadOnlySpan<ulong> maskSpan = mask is null ? ReadOnlySpan<ulong>.Empty : mask.AsSpan(0, maskLength);

        // Compute effectiveK and allocate result arrays BEFORE the expensive rotation so
        // that we short-circuit early and also catch any overflow before doing work.
        int effectiveK = GetEffectiveK(nVectors, k, maskSpan);
        float[] allScores = new float[checked(nq * effectiveK)];
        long[] allIndices = new long[checked(nq * effectiveK)];

        if (effectiveK == 0)
        {
            return (allScores, allIndices);
        }

        ReadOnlySpan<byte> subCodesSpan = subCodes.AsSpan(0, checked(nBlocks * (dim / 4) * Constants.Block));
        ReadOnlySpan<byte> plane2BlockedSpan = plane2Blocked.AsSpan(0, checked(nBlocks * (dim / 8) * Constants.Block));
        ReadOnlySpan<float> vecScalesSpan = vecScales.AsSpan(0, nVectors);
        ReadOnlySpan<float> centroidsSpan = centroids;

        int rotSize = checked(nq * dim);
        float[]? rotBuf = ArrayPool<float>.Shared.Rent(rotSize);
        try
        {
            if (rotationPlan is null)
            {
                FillRotatedQueries(queries, nq, dim, rotation, rotBuf.AsSpan(0, rotSize));
            }
            else
            {
                FillRotatedQueries(queries, nq, dim, rotationPlan, rotBuf.AsSpan(0, rotSize));
            }
            float[]? biasCorrections = ApplyQueryCalibration(
                rotBuf, nq, dim, tqplusShift, tqplusScale);
            float[] rotatedQueries = rotBuf; // alias; Parallel.For is synchronous so safe to return after

            // Collect results into per-query local arrays inside the parallel loop to avoid
            // false sharing on allScores/allIndices when effectiveK is small.
            var perQueryScores = new float[nq][];
            var perQueryIndices = new int[nq][];
            int lutSize = checked(dim * 8);

            // Avoid Parallel.For scheduling overhead for single-query searches.
            if (nq == 1)
            {
                byte[] lut = ArrayPool<byte>.Shared.Rent(lutSize);
                try
                {
                    var qRot = new ReadOnlySpan<float>(rotatedQueries, 0, dim);
                    var lutSpan = lut.AsSpan(0, lutSize);
                    var (scale, bias) = BuildLut3BitInto(qRot, centroidsSpan, lutSpan);
                    if (biasCorrections is not null)
                    {
                        bias += biasCorrections[0];
                    }

                    var (scores, indices) = Score3Bit(subCodesSpan, plane2BlockedSpan, lut, scale, bias, vecScalesSpan, dim, nVectors, nBlocks, effectiveK, maskSpan);
                    perQueryScores[0] = scores;
                    perQueryIndices[0] = indices;
                }
                finally
                {
                    ArrayPool<byte>.Shared.Return(lut);
                }
            }
            else
            {
                Parallel.For(0, nq, qi =>
                {
                    byte[] lut = ArrayPool<byte>.Shared.Rent(lutSize);
                    try
                    {
                        var qRot = new ReadOnlySpan<float>(rotatedQueries, qi * dim, dim);
                        var lutSpan = lut.AsSpan(0, lutSize);
                        var qMask = mask is null ? ReadOnlySpan<ulong>.Empty : mask.AsSpan(0, maskLength);
                        var (scale, bias) = BuildLut3BitInto(qRot, centroids, lutSpan);
                        if (biasCorrections is not null)
                        {
                            bias += biasCorrections[qi];
                        }

                        var (scores, indices) = Score3Bit(subCodes, plane2Blocked, lut, scale, bias, vecScales, dim, nVectors, nBlocks, effectiveK, qMask);
                        perQueryScores[qi] = scores;
                        perQueryIndices[qi] = indices;
                    }
                    finally
                    {
                        ArrayPool<byte>.Shared.Return(lut);
                    }
                });
            }

            for (int qi = 0; qi < nq; qi++)
            {
                Array.Copy(perQueryScores[qi], 0, allScores, qi * effectiveK, effectiveK);
                int[] qIdx = perQueryIndices[qi];
                for (int i = 0; i < effectiveK; i++)
                {
                    allIndices[(qi * effectiveK) + i] = qIdx[i];
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rotBuf);
        }

        return (allScores, allIndices);
    }

    private static void FillRotatedQueries(ReadOnlySpan<float> queries, int nq, int dim, ReadOnlySpan<float> rotation, Span<float> dest)
    {
        // Transpose outer/inner: iterate rotation rows in the outer loop so each row
        // is read once for all queries, improving cache utilization for batch searches.
        for (int row = 0; row < dim; row++)
        {
            var rotRow = rotation.Slice(row * dim, dim);
            for (int i = 0; i < nq; i++)
            {
                dest[i * dim + row] = TensorPrimitives.Dot(queries.Slice(i * dim, dim), rotRow);
            }
        }
    }

    private static void FillRotatedQueries(ReadOnlySpan<float> queries, int nq, int dim, Rotation rotation, Span<float> dest)
    {
        if (queries.Length != checked(nq * dim) || dest.Length < checked(nq * dim))
        {
            throw new ArgumentException("queries and destination lengths must match nq * dim.");
        }

        float[] scratch = ArrayPool<float>.Shared.Rent(dim);
        try
        {
            for (int i = 0; i < nq; i++)
            {
                Span<float> rotated = dest.Slice(i * dim, dim);
                queries.Slice(i * dim, dim).CopyTo(rotated);
                rotation.Apply(rotated, scratch.AsSpan(0, dim));
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(scratch);
        }
    }

    private static float[]? ApplyQueryCalibration(
        float[] rotatedQueries,
        int nq,
        int dim,
        float[]? shift,
        float[]? scale)
    {
        if (shift is null && scale is null)
        {
            return null;
        }

        if (shift is null || scale is null)
        {
            throw new ArgumentException("Calibration shift and scale must both be provided.");
        }

        ValidateCalibration(dim, shift, scale);
        if (shift.Length == 0)
        {
            return null;
        }

        float[] biasCorrections = new float[nq];
        for (int qi = 0; qi < nq; qi++)
        {
            int rowStart = qi * dim;
            double bias = 0.0;
            for (int d = 0; d < dim; d++)
            {
                float original = rotatedQueries[rowStart + d];
                rotatedQueries[rowStart + d] = original / scale[d];
                bias -= (double)original * shift[d];
            }

            biasCorrections[qi] = (float)bias;
        }

        return biasCorrections;
    }

    private static void ValidateCalibration(int dim, float[] shift, float[] scale)
    {
        if (shift.Length == 0 && scale.Length == 0)
        {
            return;
        }

        if (shift.Length != dim || scale.Length != dim)
        {
            throw new ArgumentException("Calibration shift and scale must both be empty or have length dim.");
        }
    }

    private static int GetNByteGroups(int bitWidth, int dim)
    {
        return bitWidth switch
        {
            2 => dim / 4,
            3 => dim / 4,
            4 => dim / 2,
            _ => throw new NotSupportedException("Only 2-bit, 3-bit, and 4-bit search are supported."),
        };
    }

    private static (float Scale, float Bias) BuildLut4BitInto(ReadOnlySpan<float> qRot, ReadOnlySpan<float> centroids, Span<byte> lut)
    {
        int nByteGroups = qRot.Length / 2;

        // Compute all products once, tracking min/max simultaneously, to avoid
        // iterating over centroids twice (each product is q*centroid).
        float[] products = ArrayPool<float>.Shared.Rent(nByteGroups * 32);
        float min = float.PositiveInfinity;
        float max = float.NegativeInfinity;
        float scale = 0f;
        try
        {
            for (int g = 0; g < nByteGroups; g++)
            {
                float q0 = qRot[g * 2];
                float q1 = qRot[(g * 2) + 1];
                int offset = g * 32;
                for (int c = 0; c < 16; c++)
                {
                    float v0 = q0 * centroids[c];
                    float v1 = q1 * centroids[c];
                    products[offset + c] = v0;
                    products[offset + 16 + c] = v1;
                    if (v0 < min) min = v0;
                    if (v0 > max) max = v0;
                    if (v1 < min) min = v1;
                    if (v1 > max) max = v1;
                }
            }

            scale = QuantizationScale(min, max);
            for (int i = 0; i < nByteGroups * 32; i++)
            {
                lut[i] = QuantizeToByte(products[i], min, scale);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(products);
        }

        return (scale, min * qRot.Length);
    }

    private static (float Scale, float Bias) BuildLut2BitInto(ReadOnlySpan<float> qRot, ReadOnlySpan<float> centroids, Span<byte> lut)
    {
        int nByteGroups = qRot.Length / 4;

        // Compute all products once, tracking min/max simultaneously.
        float[] products = ArrayPool<float>.Shared.Rent(nByteGroups * 16);
        float min = float.PositiveInfinity;
        float max = float.NegativeInfinity;
        float scale = 0f;
        try
        {
            for (int g = 0; g < nByteGroups; g++)
            {
                int dimOffset = g * 4;
                int offset = g * 16;
                for (int pos = 0; pos < 4; pos++)
                {
                    float q = qRot[dimOffset + pos];
                    for (int c = 0; c < 4; c++)
                    {
                        float v = q * centroids[c];
                        products[offset + (pos * 4) + c] = v;
                        if (v < min) min = v;
                        if (v > max) max = v;
                    }
                }
            }

            scale = QuantizationScale(min, max);
            for (int i = 0; i < nByteGroups * 16; i++)
            {
                lut[i] = QuantizeToByte(products[i], min, scale);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(products);
        }

        return (scale, min * qRot.Length);
    }

    private static (float Scale, float Bias) BuildLut3BitInto(ReadOnlySpan<float> qRot, ReadOnlySpan<float> centroids, Span<byte> lut)
    {
        // Compute all products once, tracking min/max simultaneously.
        float[] products = ArrayPool<float>.Shared.Rent(qRot.Length * 8);
        float min = float.PositiveInfinity;
        float max = float.NegativeInfinity;
        float scale = 0f;
        try
        {
            for (int dimIdx = 0; dimIdx < qRot.Length; dimIdx++)
            {
                float q = qRot[dimIdx];
                int offset = dimIdx * 8;
                for (int c = 0; c < 8; c++)
                {
                    float v = q * centroids[c];
                    products[offset + c] = v;
                    if (v < min) min = v;
                    if (v > max) max = v;
                }
            }

            scale = QuantizationScale(min, max);
            for (int i = 0; i < qRot.Length * 8; i++)
            {
                lut[i] = QuantizeToByte(products[i], min, scale);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(products);
        }

        return (scale, min * qRot.Length);
    }

    private static (float[] Scores, int[] Indices) Score4Bit(
        ReadOnlySpan<byte> blockedCodes,
        byte[] lut,
        float scale,
        float bias,
        ReadOnlySpan<float> vecScales,
        int dim,
        int nVectors,
        int nBlocks,
        int k,
        ReadOnlySpan<ulong> mask)
    {
        int nByteGroups = dim / 2;
        float[] heapScores = CreateScoreBuffer(k);
        int[] heapIndices = CreateIndexBuffer(k);
        int heapSize = 0;
        long localSkipped = 0;

        for (int blockIdx = 0; blockIdx < nBlocks; blockIdx++)
        {
            int baseVec = blockIdx * Constants.Block;
            if (!BlockHasAllowed(mask, baseVec, nVectors))
            {
                localSkipped++;
                continue;
            }

            int blockOffset = blockIdx * nByteGroups * Constants.Block;
            for (int lane = 0; lane < Constants.Block; lane++)
            {
                int vecIdx = baseVec + lane;
                if (vecIdx >= nVectors)
                {
                    break;
                }

                if (!IsAllowed(mask, vecIdx))
                {
                    continue;
                }

                int acc = 0;
                for (int g = 0; g < nByteGroups; g++)
                {
                    byte packed = blockedCodes[blockOffset + (g * Constants.Block) + lane];
                    acc += lut[(g * 32) + (packed >> 4)];
                    acc += lut[(g * 32) + 16 + (packed & 0x0F)];
                }

                float score = (scale * acc) + bias;
                score *= vecScales[vecIdx];
                UpdateHeap(heapScores, heapIndices, ref heapSize, k, score, vecIdx);
            }
        }

        FinalizeTopK(heapScores, heapIndices, heapSize);
        if (localSkipped > 0) Interlocked.Add(ref _blocksSkippedByMask, localSkipped);
        return (heapScores, heapIndices);
    }

    private static (float[] Scores, int[] Indices) Score2Bit(
        ReadOnlySpan<byte> blockedCodes,
        byte[] lut,
        float scale,
        float bias,
        ReadOnlySpan<float> vecScales,
        int dim,
        int nVectors,
        int nBlocks,
        int k,
        ReadOnlySpan<ulong> mask)
    {
        int nByteGroups = dim / 4;
        float[] heapScores = CreateScoreBuffer(k);
        int[] heapIndices = CreateIndexBuffer(k);
        int heapSize = 0;
        long localSkipped = 0;

        for (int blockIdx = 0; blockIdx < nBlocks; blockIdx++)
        {
            int baseVec = blockIdx * Constants.Block;
            if (!BlockHasAllowed(mask, baseVec, nVectors))
            {
                localSkipped++;
                continue;
            }

            int blockOffset = blockIdx * nByteGroups * Constants.Block;
            for (int lane = 0; lane < Constants.Block; lane++)
            {
                int vecIdx = baseVec + lane;
                if (vecIdx >= nVectors)
                {
                    break;
                }

                if (!IsAllowed(mask, vecIdx))
                {
                    continue;
                }

                int acc = 0;
                for (int g = 0; g < nByteGroups; g++)
                {
                    byte packed = blockedCodes[blockOffset + (g * Constants.Block) + lane];
                    int lutOffset = g * 16;
                    acc += lut[lutOffset + ((packed >> 6) & 0x03)];
                    acc += lut[lutOffset + 4 + ((packed >> 4) & 0x03)];
                    acc += lut[lutOffset + 8 + ((packed >> 2) & 0x03)];
                    acc += lut[lutOffset + 12 + (packed & 0x03)];
                }

                float score = (scale * acc) + bias;
                score *= vecScales[vecIdx];
                UpdateHeap(heapScores, heapIndices, ref heapSize, k, score, vecIdx);
            }
        }

        FinalizeTopK(heapScores, heapIndices, heapSize);
        if (localSkipped > 0) Interlocked.Add(ref _blocksSkippedByMask, localSkipped);
        return (heapScores, heapIndices);
    }

    private static (float[] Scores, int[] Indices) Score3Bit(
        ReadOnlySpan<byte> subCodes,
        ReadOnlySpan<byte> plane2Blocked,
        byte[] lut,
        float scale,
        float bias,
        ReadOnlySpan<float> vecScales,
        int dim,
        int nVectors,
        int nBlocks,
        int k,
        ReadOnlySpan<ulong> mask)
    {
        int subByteGroups = dim / 4;
        int plane2ByteGroups = dim / 8;
        float[] heapScores = CreateScoreBuffer(k);
        int[] heapIndices = CreateIndexBuffer(k);
        int heapSize = 0;
        long localSkipped = 0;

        for (int blockIdx = 0; blockIdx < nBlocks; blockIdx++)
        {
            int baseVec = blockIdx * Constants.Block;
            if (!BlockHasAllowed(mask, baseVec, nVectors))
            {
                localSkipped++;
                continue;
            }

            int subBlockOffset = blockIdx * subByteGroups * Constants.Block;
            int planeBlockOffset = blockIdx * plane2ByteGroups * Constants.Block;
            for (int lane = 0; lane < Constants.Block; lane++)
            {
                int vecIdx = baseVec + lane;
                if (vecIdx >= nVectors)
                {
                    break;
                }

                if (!IsAllowed(mask, vecIdx))
                {
                    continue;
                }

                int acc = 0;
                for (int g = 0; g < subByteGroups; g += 2)
                {
                    byte sub0 = subCodes[subBlockOffset + (g * Constants.Block) + lane];
                    byte sub1 = subCodes[subBlockOffset + ((g + 1) * Constants.Block) + lane];
                    byte plane2 = plane2Blocked[planeBlockOffset + ((g >> 1) * Constants.Block) + lane];
                    int dimBase0 = g * 4;
                    int dimBase1 = (g + 1) * 4;

                    for (int pos = 0; pos < 4; pos++)
                    {
                        int lowBits0 = (sub0 >> ((3 - pos) * 2)) & 0x03;
                        int highBit0 = (plane2 >> (7 - pos)) & 0x01;
                        acc += lut[(dimBase0 + pos) * 8 + (lowBits0 | (highBit0 << 2))];
                    }

                    for (int pos = 0; pos < 4; pos++)
                    {
                        int lowBits1 = (sub1 >> ((3 - pos) * 2)) & 0x03;
                        int highBit1 = (plane2 >> (3 - pos)) & 0x01;
                        acc += lut[(dimBase1 + pos) * 8 + (lowBits1 | (highBit1 << 2))];
                    }
                }

                float score = (scale * acc) + bias;
                score *= vecScales[vecIdx];
                UpdateHeap(heapScores, heapIndices, ref heapSize, k, score, vecIdx);
            }
        }

        FinalizeTopK(heapScores, heapIndices, heapSize);
        if (localSkipped > 0) Interlocked.Add(ref _blocksSkippedByMask, localSkipped);
        return (heapScores, heapIndices);
    }

    private static void UpdateHeap(float[] heapScores, int[] heapIndices, ref int heapSize, int capacity, float score, int index)
    {
        if (capacity == 0)
        {
            return;
        }

        if (heapSize < capacity)
        {
            heapScores[heapSize] = score;
            heapIndices[heapSize] = index;
            SiftUp(heapScores, heapIndices, heapSize);
            heapSize++;
            return;
        }

        if (score <= heapScores[0])
        {
            return;
        }

        heapScores[0] = score;
        heapIndices[0] = index;
        SiftDown(heapScores, heapIndices, 0, heapSize);
    }

    private static void SiftUp(float[] heapScores, int[] heapIndices, int index)
    {
        while (index > 0)
        {
            int parent = (index - 1) >> 1;
            if (!IsWorseCandidate(heapScores, heapIndices, index, parent))
            {
                break;
            }

            Swap(heapScores, heapIndices, parent, index);
            index = parent;
        }
    }

    private static void SiftDown(float[] heapScores, int[] heapIndices, int index, int heapSize)
    {
        while (true)
        {
            int left = (index * 2) + 1;
            if (left >= heapSize)
            {
                return;
            }

            int right = left + 1;
            int worst = left;
            if (right < heapSize && IsWorseCandidate(heapScores, heapIndices, right, left))
            {
                worst = right;
            }

            if (IsWorseCandidate(heapScores, heapIndices, index, worst))
            {
                return;
            }

            Swap(heapScores, heapIndices, index, worst);
            index = worst;
        }
    }

    private static bool IsWorseCandidate(float[] scores, int[] indices, int candidate, int other)
        => scores[candidate] < scores[other]
            || (scores[candidate] == scores[other] && indices[candidate] > indices[other]);

    private static void FinalizeTopK(float[] scores, int[] indices, int heapSize)
    {
        if (heapSize <= 1)
        {
            return;
        }

        for (int start = (heapSize >> 1) - 1; start >= 0; start--)
        {
            SiftDownForResultSort(scores, indices, start, heapSize);
        }

        for (int end = heapSize - 1; end > 0; end--)
        {
            Swap(scores, indices, 0, end);
            SiftDownForResultSort(scores, indices, 0, end);
        }

        scores.AsSpan(0, heapSize).Reverse();
        indices.AsSpan(0, heapSize).Reverse();
    }

    private static void SiftDownForResultSort(float[] scores, int[] indices, int index, int length)
    {
        while (true)
        {
            int left = (index * 2) + 1;
            if (left >= length)
            {
                return;
            }

            int right = left + 1;
            int greater = left;
            if (right < length && CompareForAscendingResultOrder(scores, indices, right, left) > 0)
            {
                greater = right;
            }

            if (CompareForAscendingResultOrder(scores, indices, index, greater) >= 0)
            {
                return;
            }

            Swap(scores, indices, index, greater);
            index = greater;
        }
    }

    private static int CompareForAscendingResultOrder(float[] scores, int[] indices, int left, int right)
    {
        int scoreOrder = scores[left].CompareTo(scores[right]);
        return scoreOrder != 0 ? scoreOrder : indices[right].CompareTo(indices[left]);
    }

    private static void Swap(float[] heapScores, int[] heapIndices, int a, int b)
    {
        (heapScores[a], heapScores[b]) = (heapScores[b], heapScores[a]);
        (heapIndices[a], heapIndices[b]) = (heapIndices[b], heapIndices[a]);
    }

    private static float[] CreateScoreBuffer(int length)
    {
        float[] scores = new float[length];
        Array.Fill(scores, float.NegativeInfinity);
        return scores;
    }

    private static int[] CreateIndexBuffer(int length)
    {
        int[] indices = new int[length];
        Array.Fill(indices, -1);
        return indices;
    }

    private static int GetEffectiveK(int nVectors, int k, ReadOnlySpan<ulong> mask)
    {
        int effectiveK = Math.Min(k, nVectors);
        if (mask.IsEmpty)
        {
            return effectiveK;
        }

        int wordCount = (nVectors + 63) >> 6;
        int count = 0;
        for (int word = 0; word < wordCount; word++)
        {
            ulong value = mask[word];
            if (word == wordCount - 1)
            {
                int validBits = nVectors - (word * 64);
                if (validBits < 64)
                {
                    value &= (1UL << validBits) - 1UL;
                }
            }

            count += BitOperations.PopCount(value);
            if (count >= effectiveK)
            {
                break;
            }
        }

        return Math.Min(effectiveK, count);
    }

    private static bool IsAllowed(ReadOnlySpan<ulong> mask, int index)
    {
        return mask.IsEmpty || ((mask[index >> 6] >> (index & 63)) & 1UL) != 0;
    }

    private static bool BlockHasAllowed(ReadOnlySpan<ulong> mask, int baseVec, int nVectors)
    {
        if (mask.IsEmpty)
        {
            return true;
        }

        int endExclusive = Math.Min(baseVec + Constants.Block, nVectors);
        if (baseVec >= endExclusive)
        {
            return false;
        }

        int startWord = baseVec >> 6;
        int endWord = (endExclusive - 1) >> 6;

        if (startWord == endWord)
        {
            ulong mask64 = MakeBitRangeMask(baseVec & 63, ((endExclusive - 1) & 63) + 1);
            return (mask[startWord] & mask64) != 0;
        }

        // Block spans multiple 64-bit words — check each word the block touches.
        ulong firstMask = MakeBitRangeMask(baseVec & 63, 64);
        if ((mask[startWord] & firstMask) != 0)
        {
            return true;
        }

        for (int w = startWord + 1; w < endWord; w++)
        {
            if (mask[w] != 0)
            {
                return true;
            }
        }

        ulong lastMask = MakeBitRangeMask(0, ((endExclusive - 1) & 63) + 1);
        return (mask[endWord] & lastMask) != 0;
    }

    private static ulong MakeBitRangeMask(int fromBit, int toBitExclusive)
    {
        ulong highMask = toBitExclusive == 64 ? ulong.MaxValue : ((1UL << toBitExclusive) - 1UL);
        ulong lowMask = fromBit == 0 ? 0UL : ((1UL << fromBit) - 1UL);
        return highMask & ~lowMask;
    }

    private static byte QuantizeToByte(float value, float min, float scale)
    {
        if (scale <= 1e-12f)
        {
            return 0;
        }

        int quantized = (int)MathF.Round((value - min) / scale);
        return (byte)Math.Clamp(quantized, 0, 255);
    }

    private static float QuantizationScale(float min, float max)
    {
        float range = max - min;
        return range > 1e-12f ? range / 255.0f : 1.0f;
    }

    private static void ValidateInputs(
        ReadOnlySpan<float> queries,
        int nq,
        ReadOnlySpan<float> rotation,
        ReadOnlySpan<float> centroids,
        ReadOnlySpan<float> vecScales,
        int bitWidth,
        int dim,
        int nVectors,
        int k,
        ReadOnlySpan<ulong> mask)
    {
        ValidateInputs(queries, nq, centroids, vecScales, bitWidth, dim, nVectors, k, mask);
        if (rotation.Length != checked(dim * dim))
        {
            throw new ArgumentException("rotation length must equal dim * dim.", nameof(rotation));
        }
    }

    private static void ValidateInputs(
        ReadOnlySpan<float> queries,
        int nq,
        ReadOnlySpan<float> centroids,
        ReadOnlySpan<float> vecScales,
        int bitWidth,
        int dim,
        int nVectors,
        int k,
        ReadOnlySpan<ulong> mask)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(nq);
        InputValidation.ValidateDimension(dim, nameof(dim));
        InputValidation.ValidateBitWidth(bitWidth, nameof(bitWidth));

        ArgumentOutOfRangeException.ThrowIfNegative(nVectors);
        ArgumentOutOfRangeException.ThrowIfNegative(k);

        if (queries.Length != checked(nq * dim))
        {
            throw new ArgumentException("queries length must equal nq * dim.", nameof(queries));
        }

        InputValidation.ValidateValues(queries, dim, nameof(queries));

        int expectedCentroids = 1 << bitWidth;
        if (centroids.Length != expectedCentroids)
        {
            throw new ArgumentException($"Expected centroids length {expectedCentroids}, got {centroids.Length}.", nameof(centroids));
        }

        if (vecScales.Length != nVectors)
        {
            throw new ArgumentException($"Expected vecScales length {nVectors}, got {vecScales.Length}.", nameof(vecScales));
        }

        int requiredMaskWords = (nVectors + 63) >> 6;
        if (!mask.IsEmpty && mask.Length < requiredMaskWords)
        {
            throw new ArgumentException($"Expected mask length at least {requiredMaskWords}, got {mask.Length}.", nameof(mask));
        }
    }
}
