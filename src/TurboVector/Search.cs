using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace TurboVector;

/// <summary>
/// Quantized similarity search over blocked codebooks.
/// </summary>
public static partial class Search
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
        byte[] blockedCodes,
        float[] centroids,
        float[] vecScales,
        int bitWidth,
        int dim,
        int nVectors,
        int nBlocks,
        int k,
        ulong[]? mask,
        int maskLength)
        => Search_(queries, nq, rotation, blockedCodes, centroids, vecScales, bitWidth,
            dim, nVectors, nBlocks, k, mask, maskLength, [], []);

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
        float[] tqplusScale,
        PlanesCache? planes = null)
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

        int nByteGroups = GetNByteGroups(bitWidth, dim);
        int expectedLength = checked(nBlocks * nByteGroups * Constants.Block);
        if (planes is null && blockedCodes.Length != expectedLength)
        {
            throw new ArgumentException($"Expected blockedCodes length {expectedLength}, got {blockedCodes.Length}.", nameof(blockedCodes));
        }

        return RunBlockedNibbleSearch(
            queries, nq, default, blockedCodes, centroids, vecScales, bitWidth,
            dim, nVectors, nBlocks, k, mask, maskLength, rotation, tqplusShift, tqplusScale, planes);
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
        ValidateCalibration(dim, tqplusShift, tqplusScale);

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
            maskLength,
            rotationPlan: null,
            tqplusShift: tqplusShift,
            tqplusScale: tqplusScale);
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

        var (blocked, nBlocks) = Pack.Repack(packedCodes, nVectors, 3, dim);
        return RunBlockedNibbleSearch(
            queries,
            nq,
            rotation,
            blocked,
            centroids.ToArray(),
            vecScales.ToArray(),
            3,
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
        float[]? tqplusScale = null,
        PlanesCache? planes = null)
    {
        ValidateBlockCoverage(nVectors, nBlocks);
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
        var blockedSpan = planes is null
            ? blockedCodes.AsSpan(0, checked(nBlocks * nByteGroups * Constants.Block))
            : Span<byte>.Empty;
        ReadOnlySpan<float> vecScalesSpan = vecScales.AsSpan(0, nVectors);
        ReadOnlySpan<float> centroidsSpan = centroids;
        if (planes is null && UsePlanes(bitWidth, nByteGroups, nVectors))
        {
            planes = BuildPlanesCache(blockedSpan, centroidsSpan, tqplusShift, tqplusScale,
                bitWidth, dim, nVectors, nBlocks);
        }

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
            var vectorMajor = UseVectorMajor(bitWidth, nByteGroups);
            var planeThreads = planes is null ? 1 : UpstreamThreadCount();
            var arm64 = AdvSimd.Arm64.IsSupported;

            // Avoid Parallel.For scheduling overhead for single-query searches.
            if (nq == 1)
            {
                var qRot = new ReadOnlySpan<float>(rotatedQueries, 0, dim);
                var (scores, indices) = planes is null
                    ? ScoreBlockedQuery(qRot, blockedSpan, centroidsSpan, vecScalesSpan, bitWidth, nByteGroups,
                        nVectors, nBlocks, effectiveK, maskSpan, biasCorrections?[0] ?? 0f, vectorMajor, false)
                    : ScorePlanesQuery(qRot, centroidsSpan, vecScalesSpan, planes, bitWidth,
                        nVectors, nBlocks, effectiveK, maskSpan, biasCorrections?[0] ?? 0f,
                        tqplusShift, tqplusScale, nq, planeThreads, arm64);
                perQueryScores[0] = scores;
                perQueryIndices[0] = indices;
            }
            else
            {
                var shortQueries = planes is not null && maskSpan.IsEmpty ? new bool[nq] : null;
                Parallel.For(0, nq, qi =>
                {
                    var qRot = new ReadOnlySpan<float>(rotatedQueries, qi * dim, dim);
                    var qMask = mask is null ? ReadOnlySpan<ulong>.Empty : mask.AsSpan(0, maskLength);
                    var (scores, indices) = planes is null
                        ? ScoreBlockedQuery(qRot, blockedCodes, centroids, vecScales, bitWidth, nByteGroups,
                            nVectors, nBlocks, effectiveK, qMask, biasCorrections?[qi] ?? 0f, vectorMajor,
                            bitWidth == 2 && IsArmPermuteBatch(qi, nq) && UseArm2BitPermuteDot(nByteGroups))
                        : ScorePlanesQuery(qRot, centroids, vecScales, planes, bitWidth,
                            nVectors, nBlocks, effectiveK, qMask, biasCorrections?[qi] ?? 0f,
                            tqplusShift, tqplusScale, nq, planeThreads, arm64, shortQueries, qi);
                    perQueryScores[qi] = scores;
                    perQueryIndices[qi] = indices;
                });

                if (shortQueries is not null)
                {
                    var rescanQueries = 0;
                    foreach (var isShort in shortQueries)
                    {
                        if (isShort) rescanQueries++;
                    }

                    if (rescanQueries > 0)
                    {
                        Parallel.For(0, nq, qi =>
                        {
                            if (!shortQueries[qi]) return;
                            var qRot = new ReadOnlySpan<float>(rotatedQueries, qi * dim, dim);
                            var (scores, indices) = ScorePlanesQuery(qRot, centroids, vecScales, planes!, bitWidth,
                                nVectors, nBlocks, effectiveK, [], biasCorrections?[qi] ?? 0f,
                                tqplusShift, tqplusScale, nq, planeThreads, arm64, shortQueries, qi, rescanQueries);
                            perQueryScores[qi] = scores;
                            perQueryIndices[qi] = indices;
                        });
                    }
                }
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
        ValidateBlockCoverage(nVectors, nBlocks);
        var maskSpan = mask is null ? ReadOnlySpan<ulong>.Empty : mask.AsSpan(0, maskLength);
        if (nq == 0 || GetEffectiveK(nVectors, k, maskSpan) == 0) return ([], []);

        var length = checked(nBlocks * (dim / 2) * Constants.Block);
        var blocked = ArrayPool<byte>.Shared.Rent(length);
        try
        {
            Pack.Expand3BitBlocked(subCodes, plane2Blocked, dim, nBlocks, blocked.AsSpan(0, length));
            return RunBlockedNibbleSearch(
                queries, nq, rotation, blocked, centroids, vecScales, 3, dim,
                nVectors, nBlocks, k, mask, maskLength, rotationPlan, tqplusShift, tqplusScale);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(blocked);
        }
    }

    private static void ValidateBlockCoverage(int nVectors, int nBlocks)
    {
        var requiredBlocks = nVectors / Constants.Block + (nVectors % Constants.Block == 0 ? 0 : 1);
        if (nBlocks < requiredBlocks)
        {
            throw new ArgumentOutOfRangeException(nameof(nBlocks), $"At least {requiredBlocks} blocks are required for {nVectors} vectors.");
        }
    }

    internal static (float[] Scores, int[] Indices) ScoreBlockedQuery(
        ReadOnlySpan<float> query,
        ReadOnlySpan<byte> blockedCodes,
        ReadOnlySpan<float> centroids,
        ReadOnlySpan<float> vecScales,
        int bitWidth,
        int nByteGroups,
        int nVectors,
        int nBlocks,
        int k,
        ReadOnlySpan<ulong> mask,
        float biasCorrection,
        bool vectorMajor,
        bool arm2BitPermuteDot = false)
    {
        if (arm2BitPermuteDot)
        {
            return Score2BitPermuteDot(query, blockedCodes, centroids, vecScales, nByteGroups,
                nVectors, nBlocks, k, mask, biasCorrection);
        }

        if (vectorMajor && bitWidth == 4)
        {
            return ScorePermuteDot(query, blockedCodes, centroids, vecScales, nByteGroups,
                nVectors, nBlocks, k, mask, biasCorrection);
        }

        var lutSize = checked(nByteGroups * 32);
        var lut = ArrayPool<byte>.Shared.Rent(lutSize);
        try
        {
            var (scale, bias) = bitWidth switch
            {
                2 => BuildLut2BitInto(query, centroids, lut.AsSpan(0, lutSize)),
                3 => BuildLut3BitInto(query, centroids, lut.AsSpan(0, lutSize)),
                4 => BuildLut4BitInto(query, centroids, lut.AsSpan(0, lutSize)),
                _ => throw new NotSupportedException("Only 2-bit, 3-bit, and 4-bit search are supported."),
            };
            bias += biasCorrection;
            return ScoreNibble(blockedCodes, lut, scale, bias, vecScales, nByteGroups,
                nVectors, nBlocks, k, mask, vectorMajor);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(lut);
        }
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
            3 => dim / 2,
            4 => dim / 2,
            _ => throw new NotSupportedException("Only 2-bit, 3-bit, and 4-bit search are supported."),
        };
    }

    private static (float Scale, float Bias) BuildLut4BitInto(ReadOnlySpan<float> qRot, ReadOnlySpan<float> centroids, Span<byte> lut)
        => BuildSingleCodeLutInto(qRot, centroids, lut);

    private static (float Scale, float Bias) BuildLut3BitInto(ReadOnlySpan<float> qRot, ReadOnlySpan<float> centroids, Span<byte> lut)
        => BuildSingleCodeLutInto(qRot, centroids, lut);

    private static (float Scale, float Bias) BuildSingleCodeLutInto(ReadOnlySpan<float> qRot, ReadOnlySpan<float> centroids, Span<byte> lut)
    {
        var levels = centroids.Length;
        var products = ArrayPool<float>.Shared.Rent(checked(qRot.Length * levels));
        var minima = ArrayPool<float>.Shared.Rent(qRot.Length);
        var maxSpan = 0f;
        var bias = 0f;
        try
        {
            for (var group = 0; group < qRot.Length / 2; group++)
            {
                for (var half = 0; half < 2; half++)
                {
                    var dim = group * 2 + half;
                    var min = float.MaxValue;
                    var max = float.MinValue;
                    for (var code = 0; code < levels; code++)
                    {
                        var value = 0f + qRot[dim] * centroids[code];
                        products[dim * levels + code] = value;
                        if (value < min) min = value;
                        if (value > max) max = value;
                    }

                    minima[dim] = min;
                    var span = max - min;
                    if (span > maxSpan) maxSpan = span;
                }

                bias += minima[group * 2] + minima[group * 2 + 1];
            }

            var scale = QuantizationScale(maxSpan);
            var inverseScale = 1f / scale;
            for (var dim = 0; dim < qRot.Length; dim++)
            {
                for (var code = 0; code < levels; code++)
                {
                    var offset = dim * levels + code;
                    var value = QuantizeToByte(products[offset], minima[dim], inverseScale);
                    lut[dim * 16 + code] = value;
                    if (levels == 8) lut[dim * 16 + 8 + code] = value;
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

    private static (float Scale, float Bias) BuildLut2BitInto(ReadOnlySpan<float> qRot, ReadOnlySpan<float> centroids, Span<byte> lut)
    {
        var groups = qRot.Length / 4;
        var products = ArrayPool<float>.Shared.Rent(checked(groups * 32));
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
                    var dim = group * 4 + half * 2;
                    for (var code = 0; code < 4; code++)
                    {
                        first[code] = 0f + qRot[dim] * centroids[code];
                        second[code] = qRot[dim + 1] * centroids[code];
                    }

                    var minFirst = first[0];
                    var maxFirst = first[0];
                    var minSecond = second[0];
                    var maxSecond = second[0];
                    for (var code = 1; code < 4; code++)
                    {
                        if (first[code] < minFirst) minFirst = first[code];
                        if (first[code] > maxFirst) maxFirst = first[code];
                        if (second[code] < minSecond) minSecond = second[code];
                        if (second[code] > maxSecond) maxSecond = second[code];
                    }

                    var offset = group * 32 + half * 16;
                    for (var a = 0; a < 4; a++)
                    {
                        for (var b = 0; b < 4; b++)
                        {
                            products[offset + a * 4 + b] = first[a] + second[b];
                        }
                    }

                    var min = minFirst + minSecond;
                    minima[group * 2 + half] = min;
                    var span = (maxFirst + maxSecond) - min;
                    if (span > maxSpan) maxSpan = span;
                }

                bias += minima[group * 2] + minima[group * 2 + 1];
            }

            var scale = QuantizationScale(maxSpan);
            var inverseScale = 1f / scale;
            for (var table = 0; table < groups * 2; table++)
            {
                for (var code = 0; code < 16; code++)
                {
                    var offset = table * 16 + code;
                    lut[offset] = QuantizeToByte(products[offset], minima[table], inverseScale);
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

    private static (float[] Scores, int[] Indices) ScoreNibble(
        ReadOnlySpan<byte> blockedCodes,
        byte[] lut,
        float scale,
        float bias,
        ReadOnlySpan<float> vecScales,
        int nByteGroups,
        int nVectors,
        int nBlocks,
        int k,
        ReadOnlySpan<ulong> mask,
        bool vectorMajor)
    {
        if (Avx2.IsSupported && Fma.IsSupported)
        {
            return ScoreNibbleAvx2(blockedCodes, lut, scale, bias, vecScales, nByteGroups, nVectors, nBlocks, k, mask, vectorMajor);
        }

        if (AdvSimd.Arm64.IsSupported)
        {
            return ScoreNibbleArm64(blockedCodes, lut, scale, bias, vecScales, nByteGroups, nVectors, nBlocks, k, mask);
        }

        var heapScores = CreateScoreBuffer(k);
        var heapIndices = CreateIndexBuffer(k);
        var heapSize = 0;
        var localSkipped = 0L;

        for (var blockIdx = 0; blockIdx < nBlocks; blockIdx++)
        {
            var baseVec = blockIdx * Constants.Block;
            if (!BlockHasAllowed(mask, baseVec, nVectors))
            {
                localSkipped++;
                continue;
            }

            var blockOffset = blockIdx * nByteGroups * Constants.Block;
            for (var lane = 0; lane < Constants.Block; lane++)
            {
                var vecIdx = baseVec + lane;
                if (vecIdx >= nVectors)
                {
                    break;
                }

                if (!IsAllowed(mask, vecIdx))
                {
                    continue;
                }

                var score = bias;
                for (var group = 0; group < nByteGroups; group++)
                {
                    var packed = blockedCodes[blockOffset + group * Constants.Block + lane];
                    score += scale * lut[group * 32 + (packed >> 4)];
                    score += scale * lut[group * 32 + 16 + (packed & 0x0F)];
                }

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

    private static byte QuantizeToByte(float value, float min, float inverseScale, byte cap = 127)
    {
        var quantized = (value - min) * inverseScale;
        if (!(quantized > 0f)) return 0;
        if (quantized >= cap) return cap;
        return (byte)MathF.Round(quantized, MidpointRounding.AwayFromZero);
    }

    private static float QuantizationScale(float maxSpan)
    {
        var scale = maxSpan > 0f ? maxSpan / 127f : 1f;
        return scale >= 1.17549435e-38f ? scale : 1.0f;
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
