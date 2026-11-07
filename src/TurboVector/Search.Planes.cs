using System.Buffers;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace TurboVector;

public static partial class Search
{
    private static readonly Lazy<int> PlaneThreadCount = new(() =>
    {
        var value = Environment.GetEnvironmentVariable("RAYON_NUM_THREADS");
        return int.TryParse(value, out var threads) && threads > 0 ? threads : Environment.ProcessorCount;
    });

    internal static (float[] Scores, int[] Indices) ScorePlanesQuery(
        ReadOnlySpan<float> query,
        ReadOnlySpan<float> centroids,
        ReadOnlySpan<float> vectorScales,
        PlanesCache cache,
        int bits,
        int nVectors,
        int nBlocks,
        int k,
        ReadOnlySpan<ulong> mask,
        float biasCorrection,
        ReadOnlySpan<float> shift,
        ReadOnlySpan<float> calibrationScale,
        int queryCount,
        int threads,
        bool arm64,
        bool[]? shortQueries = null,
        int queryIndex = 0,
        int rescanQueries = 0)
    {
        var dim = query.Length;
        var signGroups = dim / 8;
        var groups = dim / (8 / bits);
        nBlocks = Math.Min(nBlocks, nVectors / Constants.Block + (nVectors % Constants.Block == 0 ? 0 : 1));
        var allowed = GetEffectiveK(nVectors, nVectors, mask);
        var shortlist = (int)Math.Min(int.MaxValue,
            Math.Max(bits == 2 ? 384L : 256L, (long)k * (bits == 4 && k < 64 ? 20 : 16)));
        var rescore = (int)Math.Min(int.MaxValue,
            Math.Max(64L, bits == 2 ? (long)k * 2 : (long)k * 3 / 2));
        var middle = (int)Math.Min(int.MaxValue, Math.Max(256L, (long)k * 6));
        var poolRescoresAll = !arm64 && queryCount == 1 && threads > 1
            && CountAllowedBlocks(mask, nVectors, nBlocks) >= 1024 && shortlist < 640;
        var refine = shortlist < allowed && rescore < shortlist && !poolRescoresAll;
        var inRangeRefine = queryCount == 1 && threads > 1
            && (arm64 || shortlist >= 640) && refine && mask.IsEmpty
            && nBlocks >= 1024 && 2L * shortlist < nVectors;
        var cap = (byte)(arm64 && queryCount == 1 ? 31 : 127);
        var lut = ArrayPool<byte>.Shared.Rent(checked(signGroups * 32));
        var signQueryBuffer = cache.Centre is null ? null : ArrayPool<float>.Shared.Rent(dim);
        var rescanLut = default(byte[]);
        try
        {
            var signQuery = query;
            if (signQueryBuffer is not null)
            {
                for (var d = 0; d < dim; d++)
                {
                    signQueryBuffer[d] = query[d] + 0.5f * shift[d] / calibrationScale[d];
                }
                signQuery = signQueryBuffer.AsSpan(0, dim);
            }

            var (signScale, signBias) = BuildSignLut(signQuery, cache.Magnitude, cap, lut);
            signBias += biasCorrection;
            var candidateScores = Array.Empty<float>();
            var candidateIndices = Array.Empty<int>();
            if (shortlist >= allowed)
            {
                candidateScores = new float[allowed];
                candidateIndices = new int[allowed];
                var position = 0;
                for (var v = 0; v < nVectors; v++)
                {
                    if (IsAllowed(mask, v)) candidateIndices[position++] = v;
                }
            }
            else
            {
                if (2L * shortlist < allowed)
                {
                    var scanLut = lut;
                    var scanScale = signScale;
                    var scanBias = signBias;
                    if (arm64 && rescanQueries == 1)
                    {
                        rescanLut = ArrayPool<byte>.Shared.Rent(checked(signGroups * 32));
                        scanLut = rescanLut;
                        (scanScale, scanBias) = BuildSignLut(signQuery, cache.Magnitude, 31, scanLut);
                        scanBias += biasCorrection;
                    }

                    var deferRescan = shortQueries is not null && rescanQueries == 0;
                    var (scores, indices, seedShort) = CollectBufferedPlaneRanges(
                        query, cache, scanLut, scanScale, scanBias, vectorScales,
                        bits, nVectors, nBlocks, shortlist, k, mask,
                        rescanQueries == 0 ? queryCount : rescanQueries, threads,
                        cap, biasCorrection, arm64, inRangeRefine, deferRescan, rescanQueries > 0);
                    if (seedShort && deferRescan)
                    {
                        shortQueries![queryIndex] = true;
                        return ([], []);
                    }
                    candidateScores = scores;
                    candidateIndices = indices;
                }
                else
                {
                    (candidateScores, candidateIndices) = ScoreNibble(
                        cache.Signs, lut, signScale, signBias, vectorScales, signGroups,
                        nVectors, nBlocks, shortlist, mask, !arm64);
                }
                if (!inRangeRefine && cache.Centre is not null)
                {
                    for (var i = 0; i < candidateIndices.Length; i++)
                    {
                        var v = candidateIndices[i];
                        candidateScores[i] -= 0.5f * vectorScales[v] * cache.Centre[v];
                    }
                }
            }

            if (refine && candidateIndices.Length > rescore)
            {
                var lowQuery = arm64 && bits == 4 ? null : BuildLowPlaneQuery(query, cache.Magnitude);
                var plainLut = lut;
                var plainScale = signScale;
                var plainBias = signBias;
                var rentedPlainLut = default(byte[]);
                if (arm64 && bits == 4 && cache.Centre is not null)
                {
                    rentedPlainLut = ArrayPool<byte>.Shared.Rent(signGroups * 32);
                    plainLut = rentedPlainLut;
                    (plainScale, plainBias) = BuildSignLut(query, cache.Magnitude, cap, plainLut);
                    plainBias += biasCorrection;
                }

                try
                {
                    if (!inRangeRefine)
                    {
                        RankPlanes(candidateScores, candidateIndices, vectorScales, cache, lowQuery,
                            plainLut, plainScale, plainBias - biasCorrection, biasCorrection,
                            bits, signGroups, first: true);
                    }
                    if (bits == 4)
                    {
                        KeepPlaneCandidates(ref candidateScores, ref candidateIndices, middle);
                        RankPlanes(candidateScores, candidateIndices, vectorScales, cache, lowQuery,
                            plainLut, plainScale, plainBias - biasCorrection, biasCorrection,
                            bits, signGroups, first: false);
                    }
                    KeepPlaneCandidates(ref candidateScores, ref candidateIndices, rescore);
                }
                finally
                {
                    if (rentedPlainLut is not null) ArrayPool<byte>.Shared.Return(rentedPlainLut);
                }
            }

            return RescorePlaneCandidates(query, centroids, vectorScales, cache,
                candidateIndices, bits, groups, k, biasCorrection, arm64);
        }
        finally
        {
            if (rescanLut is not null) ArrayPool<byte>.Shared.Return(rescanLut);
            ArrayPool<byte>.Shared.Return(lut);
            if (signQueryBuffer is not null) ArrayPool<float>.Shared.Return(signQueryBuffer);
        }
    }

    private static (float[] Scores, int[] Indices, bool SeedShort) CollectBufferedPlaneRanges(
        ReadOnlySpan<float> query,
        PlanesCache cache,
        byte[] signLut,
        float signScale,
        float signBias,
        ReadOnlySpan<float> vectorScales,
        int bits,
        int nVectors,
        int nBlocks,
        int shortlist,
        int k,
        ReadOnlySpan<ulong> mask,
        int queryCount,
        int threads,
        byte cap,
        float biasCorrection,
        bool arm64,
        bool inRangeRefine,
        bool deferRescan,
        bool unseededRescan)
    {
        var groups = query.Length / 8;
        const int scoreChunkBlocks = 64;
        var scoreBuffer = ArrayPool<float>.Shared.Rent(scoreChunkBlocks * Constants.Block);
        var rentedPlainLut = default(byte[]);
        try
        {
            var seed = mask.IsEmpty && !unseededRescan
                ? PlaneSampleSeed(cache, signLut, signScale, signBias, vectorScales,
                    nVectors, groups, shortlist, bits, arm64, scoreBuffer)
                : float.NegativeInfinity;
            var lowQuery = !inRangeRefine || arm64 && bits == 4 ? null : BuildLowPlaneQuery(query, cache.Magnitude);
            var plainLut = signLut;
            var plainScale = signScale;
            var plainBias = signBias;
            if (inRangeRefine && arm64 && bits == 4 && cache.Centre is not null)
            {
                rentedPlainLut = ArrayPool<byte>.Shared.Rent(groups * 32);
                plainLut = rentedPlainLut;
                (plainScale, plainBias) = BuildSignLut(query, cache.Magnitude, cap, plainLut);
                plainBias += biasCorrection;
            }

            var (scores, indices) = CollectPlaneRanges(vectorScales, cache, lowQuery,
                signLut, signScale, signBias, scoreBuffer,
                plainLut, plainScale, plainBias - biasCorrection, biasCorrection,
                bits, groups, nVectors, nBlocks, shortlist, k, mask, queryCount, threads, seed,
                arm64, inRangeRefine, unseededRescan);
            var seedShort = indices.Length < shortlist && !float.IsNegativeInfinity(seed);
            if (seedShort && !deferRescan)
            {
                (scores, indices) = CollectPlaneRanges(vectorScales, cache, lowQuery,
                    signLut, signScale, signBias, scoreBuffer,
                    plainLut, plainScale, plainBias - biasCorrection, biasCorrection,
                    bits, groups, nVectors, nBlocks, shortlist, k, mask, queryCount, threads,
                    float.NegativeInfinity, arm64, inRangeRefine, true);
            }
            return (scores, indices, seedShort);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(scoreBuffer);
            if (rentedPlainLut is not null) ArrayPool<byte>.Shared.Return(rentedPlainLut);
        }
    }

    private static float PlaneSampleSeed(
        PlanesCache cache, byte[] lut, float scale, float bias, ReadOnlySpan<float> vectorScales,
        int nVectors, int groups, int shortlist, int bits, bool arm64, float[] scoreBuffer)
    {
        const int sampleBlocks = 48;
        const int sampleCount = sampleBlocks * Constants.Block;
        var fullBlocks = nVectors / Constants.Block;
        if (fullBlocks < 1024) return float.NegativeInfinity;
        var expected = (float)((long)shortlist * sampleCount) / nVectors;
        var width = bits == 4 ? 4f : 6f;
        var over = MathF.Min(1f + width / MathF.Sqrt(MathF.Max(expected, 1f)), 4f);
        var rank = Math.Clamp((int)MathF.Ceiling(over * expected), 6, sampleCount);
        var bestScores = CreateScoreBuffer(rank);
        var bestIndices = CreateIndexBuffer(rank);
        var count = 0;
        var stride = fullBlocks / sampleBlocks;
        for (var sample = 0; sample < sampleBlocks; sample++)
        {
            var block = sample * stride + stride / 2;
            var start = block * Constants.Block;
            ScorePlaneBlocks(cache, lut, scale, bias, vectorScales, nVectors, groups, block, 1, arm64, scoreBuffer);
            for (var lane = 0; lane < Constants.Block; lane++)
            {
                var index = start + lane;
                UpdateHeapWithTies(bestScores, bestIndices, ref count, rank, scoreBuffer[lane], index);
            }
        }
        return bestScores[0];
    }

    private static (float[] Scores, int[] Indices) CollectPlaneRanges(
        ReadOnlySpan<float> vectorScales,
        PlanesCache cache,
        LowPlaneQuery? query,
        byte[] signLut,
        float signScale,
        float signBias,
        float[] scoreBuffer,
        ReadOnlySpan<byte> plainLut,
        float plainScale,
        float plainBias,
        float biasCorrection,
        int bits,
        int groups,
        int nVectors,
        int nBlocks,
        int shortlist,
        int k,
        ReadOnlySpan<ulong> mask,
        int queryCount,
        int threads,
        float seed,
        bool arm64,
        bool inRangeRefine,
        bool unseededRescan)
    {
        var rangeBlocks = PlaneRangeBlocks(
            nVectors, nBlocks, shortlist, k, mask, queryCount, threads, bits, arm64, unseededRescan);
        var merged = new List<PlaneCandidate>();
        var capacity = checked(shortlist * 2);
        var candidates = ArrayPool<PlaneCandidate>.Shared.Rent(capacity);
        try
        {
            var skipped = 0L;
            for (var blockStart = 0; blockStart < nBlocks; blockStart += rangeBlocks)
            {
                var blockEnd = Math.Min(nBlocks, blockStart + rangeBlocks);
                var count = 0;
                var threshold = seed;
                for (var chunk = blockStart; chunk < blockEnd; chunk += 64)
                {
                    var chunkEnd = Math.Min(blockEnd, chunk + 64);
                    var block = chunk;
                    while (block < chunkEnd)
                    {
                        if (!mask.IsEmpty && !BlockHasAllowed(mask, block * Constants.Block, nVectors))
                        {
                            skipped++;
                            block++;
                            continue;
                        }

                        var runStart = block;
                        if (mask.IsEmpty)
                        {
                            block = chunkEnd;
                        }
                        else
                        {
                            do { block++; }
                            while (block < chunkEnd && BlockHasAllowed(mask, block * Constants.Block, nVectors));
                        }

                        var runBlocks = block - runStart;
                        var start = runStart * Constants.Block;
                        var length = (int)Math.Min((long)runBlocks * Constants.Block, nVectors - (long)start);
                        ScorePlaneBlocks(cache, signLut, signScale, signBias, vectorScales,
                            nVectors, groups, runStart, runBlocks, arm64, scoreBuffer);
                        for (var i = 0; i < length; i++)
                        {
                            var score = scoreBuffer[i];
                            var v = start + i;
                            if (!(score > threshold) || !IsAllowed(mask, v)) continue;
                            candidates[count++] = new PlaneCandidate(score, v);
                            if (count == capacity)
                            {
                                SelectPlaneCandidate(candidates.AsSpan(0, count), shortlist - 1, tieByIndex: true);
                                threshold = candidates[shortlist - 1].Score;
                                count = shortlist;
                            }
                        }
                    }
                }
                if (inRangeRefine && count > 0)
                {
                    var scores = ArrayPool<float>.Shared.Rent(count);
                    var indices = ArrayPool<int>.Shared.Rent(count);
                    try
                    {
                        for (var i = 0; i < count; i++)
                        {
                            var candidate = candidates[i];
                            indices[i] = candidate.Index;
                            scores[i] = candidate.Score;
                            if (cache.Centre is not null)
                            {
                                scores[i] -= 0.5f * vectorScales[candidate.Index] * cache.Centre[candidate.Index];
                            }
                        }
                        RankPlanes(scores.AsSpan(0, count), indices.AsSpan(0, count), vectorScales,
                            cache, query, plainLut, plainScale, plainBias, biasCorrection, bits, groups, first: true);
                        for (var i = 0; i < count; i++) candidates[i] = new PlaneCandidate(scores[i], indices[i]);
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(scores);
                        ArrayPool<int>.Shared.Return(indices);
                    }
                }
                for (var i = 0; i < count; i++) merged.Add(candidates[i]);
            }
            if (skipped > 0) Interlocked.Add(ref _blocksSkippedByMask, skipped);

            var result = merged.ToArray();
            if (result.Length > shortlist)
            {
                SelectPlaneCandidate(result, shortlist - 1, tieByIndex: true);
            }
            var resultLength = Math.Min(result.Length, shortlist);
            var resultScores = new float[resultLength];
            var resultIndices = new int[resultLength];
            for (var i = 0; i < resultLength; i++)
            {
                resultScores[i] = result[i].Score;
                resultIndices[i] = result[i].Index;
            }
            return (resultScores, resultIndices);
        }
        finally
        {
            ArrayPool<PlaneCandidate>.Shared.Return(candidates);
        }
    }

    private static void ScorePlaneBlocks(
        PlanesCache cache, byte[] lut, float scale, float bias, ReadOnlySpan<float> vectorScales,
        int nVectors, int groups, int blockStart, int blockCount, bool arm64, float[] scores)
    {
        var start = blockStart * Constants.Block;
        var length = (int)Math.Min((long)blockCount * Constants.Block, nVectors - (long)start);
        var codes = cache.Signs.AsSpan(blockStart * groups * Constants.Block, blockCount * groups * Constants.Block);
        var scales = vectorScales.Slice(start, length);
        if (!arm64 && Avx2.IsSupported && Fma.IsSupported)
        {
            _ = ScoreNibbleAvx2(codes, lut, scale, bias, scales, groups, length, blockCount, 0, [], true, scores);
            return;
        }
        if (arm64 && AdvSimd.Arm64.IsSupported)
        {
            _ = ScoreNibbleArm64(codes, lut, scale, bias, scales, groups, length, blockCount, 0, [], scores);
            return;
        }
        for (var v = 0; v < length; v++)
        {
            var block = v / Constants.Block;
            var lane = v % Constants.Block;
            var sum = 0u;
            var value = bias;
            for (var group = 0; group < groups; group++)
            {
                var code = codes[(block * groups + group) * Constants.Block + lane];
                sum += (uint)(lut[group * 32 + (code >> 4)] + lut[group * 32 + 16 + (code & 15)]);
                if (arm64 && (group + 1) % 256 == 0)
                {
                    value = MathF.FusedMultiplyAdd(scale, sum, value);
                    sum = 0;
                }
            }
            if (arm64)
            {
                if (groups % 256 != 0) value = MathF.FusedMultiplyAdd(scale, sum, value);
            }
            else
            {
                value = sum * scale + bias;
            }
            scores[v] = value * scales[v];
        }
    }

    private static int PlaneRangeBlocks(
        int nVectors, int nBlocks, int shortlist, int k, ReadOnlySpan<ulong> mask,
        int queryCount, int threads, int bits, bool arm64, bool unseededRescan)
    {
        if (queryCount == 1)
        {
            var workers = CountAllowedBlocks(mask, nVectors, nBlocks) >= 1024 ? threads : 1;
            var pieces = workers > 1 ? 2L * workers : 1;
            var stride = (int)Math.Max(64, ((long)nBlocks + pieces - 1) / pieces);
            return (stride + 1) & ~1;
        }

        if (threads <= 1 || !arm64 && !mask.IsEmpty) return nBlocks;
        var batches = (queryCount + (arm64 ? 3 : 5)) / (arm64 ? 4 : 6);
        // Both index widths scan their sign planes through the 2-bit kernel.
        var tileFloor = arm64 ? 1024 : 3072;
        var kCap = arm64 ? bits == 4 && !unseededRescan ? k : Math.Max(1, 2 * shortlist / 25) : 2 * shortlist;
        var capByK = Math.Max(1L, ((long)nVectors + 512L * kCap - 1) / (512L * kCap));
        var ranges = Math.Max(1L, Math.Min(
            Math.Min(((long)threads * (arm64 ? 64 : 32) + batches - 1) / batches,
                ((long)nBlocks + tileFloor - 1) / tileFloor),
            capByK));
        var tiles = batches * ranges;
        if (tiles > threads && tiles < 2L * threads) ranges = Math.Max(1, threads / batches);
        return (int)(((long)nBlocks + ranges - 1) / ranges);
    }

    private static void RankPlanes(
        Span<float> scores,
        ReadOnlySpan<int> indices,
        ReadOnlySpan<float> vectorScales,
        PlanesCache cache,
        LowPlaneQuery? query,
        ReadOnlySpan<byte> lut,
        float lutScale,
        float lutBias,
        float biasCorrection,
        int bits,
        int groups,
        bool first)
    {
        var lowPlanes = bits - 1;
        var a = cache.Alpha / cache.Magnitude;
        var aFirst = cache.AlphaFirst / cache.Magnitude;
        var bFirst = cache.BetaFirst / cache.Magnitude;
        for (var i = 0; i < indices.Length; i++)
        {
            var v = indices[i];
            var vectorScale = vectorScales[v];
            if (vectorScale == 0f)
            {
                scores[i] = 0f;
                continue;
            }

            var lowBase = v * lowPlanes * groups;
            var topRow = cache.Low.AsSpan(lowBase + (lowPlanes - 1) * groups, groups);
            var top = query is null
                ? TablePlaneTerm(lut, lutScale, lutBias, topRow)
                : LowPlaneTerm(query, topRow);
            if (first)
            {
                var sign = scores[i] / vectorScale - biasCorrection;
                scores[i] = vectorScale * (aFirst * sign + bFirst * top + biasCorrection);
                continue;
            }

            var recoveredSign = ((scores[i] / vectorScale - biasCorrection) - bFirst * top) / aFirst;
            var estimate = a * recoveredSign;
            for (var plane = 0; plane < lowPlanes; plane++)
            {
                var row = cache.Low.AsSpan(lowBase + plane * groups, groups);
                var term = plane == lowPlanes - 1 ? top
                    : query is null ? TablePlaneTerm(lut, lutScale, lutBias, row) : LowPlaneTerm(query, row);
                estimate += cache.Beta[plane] / cache.Magnitude * term;
            }
            scores[i] = vectorScale * (estimate + biasCorrection);
        }
    }

    private static void KeepPlaneCandidates(ref float[] scores, ref int[] indices, int count)
    {
        if (indices.Length <= count) return;
        var candidates = ArrayPool<PlaneCandidate>.Shared.Rent(indices.Length);
        try
        {
            for (var i = 0; i < indices.Length; i++)
            {
                candidates[i] = new PlaneCandidate(scores[i], indices[i]);
            }
            SelectPlaneCandidate(candidates.AsSpan(0, indices.Length), count - 1, tieByIndex: false);
            scores = new float[count];
            indices = new int[count];
            for (var i = 0; i < count; i++)
            {
                scores[i] = candidates[i].Score;
                indices[i] = candidates[i].Index;
            }
        }
        finally
        {
            ArrayPool<PlaneCandidate>.Shared.Return(candidates);
        }
    }

    private static (float[] Scores, int[] Indices) RescorePlaneCandidates(
        ReadOnlySpan<float> query,
        ReadOnlySpan<float> centroids,
        ReadOnlySpan<float> vectorScales,
        PlanesCache cache,
        ReadOnlySpan<int> candidates,
        int bits,
        int groups,
        int k,
        float biasCorrection,
        bool arm64)
    {
        var scores = CreateScoreBuffer(k);
        var indices = CreateIndexBuffer(k);
        var size = 0;
        var lut = ArrayPool<byte>.Shared.Rent(checked(groups * 32));
        var weights = bits == 4 ? ArrayPool<sbyte>.Shared.Rent(query.Length) : null;
        Span<sbyte> levels = stackalloc sbyte[16];
        try
        {
            var scale = 0f;
            var bias = biasCorrection;
            if (bits == 4)
            {
                scale = BuildPermuteOperands(query, centroids, weights!, levels);
            }
            else
            {
                (scale, bias) = BuildLut2BitInto(query, centroids, lut);
                bias += biasCorrection;
            }

            foreach (var v in candidates)
            {
                var score = bias;
                if (bits == 2)
                {
                    var sum = 0u;
                    for (var g = 0; g < groups; g++)
                    {
                        var code = ReadPlaneCode(cache, v, bits, g, query.Length / 8);
                        sum += (uint)(lut[g * 32 + (code >> 4)] + lut[g * 32 + 16 + (code & 15)]);
                        if (arm64 && (g + 1) % 256 == 0)
                        {
                            score = MathF.FusedMultiplyAdd(scale, sum, score);
                            sum = 0;
                        }
                    }
                    if (arm64)
                    {
                        if (groups % 256 != 0) score = MathF.FusedMultiplyAdd(scale, sum, score);
                    }
                    else
                    {
                        score = sum * scale + bias;
                    }
                }
                else
                {
                    var sum = 0;
                    for (var g = 0; g < groups; g++)
                    {
                        var code = ReadPlaneCode(cache, v, bits, g, query.Length / 8);
                        sum += weights![g * 2] * levels[code >> 4];
                        sum += weights[g * 2 + 1] * levels[code & 15];
                    }
                    score = MathF.FusedMultiplyAdd(sum, scale, bias);
                }
                score *= vectorScales[v];
                UpdateHeapWithTies(scores, indices, ref size, k, score, v);
            }
            FinalizeTopK(scores, indices, size);
            return (scores, indices);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(lut);
            if (weights is not null) ArrayPool<sbyte>.Shared.Return(weights);
        }
    }

    internal static byte ReadPlaneCode(PlanesCache cache, int vector, int bits, int group, int signGroups)
    {
        var planeGroup = group / bits;
        var signOffset = (vector / Constants.Block * signGroups + planeGroup) * Constants.Block
            + vector % Constants.Block;
        var lowOffset = vector * (bits - 1) * signGroups + planeGroup;
        if (bits == 2)
        {
            var shift = (1 - (group & 1)) * 4;
            var sign = (cache.Signs[signOffset] >> shift) & 15;
            var low = (cache.Low[lowOffset] >> shift) & 15;
            var spreadSign = ((sign & 8) << 3) | ((sign & 4) << 2) | ((sign & 2) << 1) | (sign & 1);
            var spreadLow = ((low & 8) << 3) | ((low & 4) << 2) | ((low & 2) << 1) | (low & 1);
            return (byte)((spreadSign << 1) | spreadLow);
        }

        var pairShift = 6 - (group & 3) * 2;
        var pairSign = (cache.Signs[signOffset] >> pairShift) & 3;
        var pair0 = (cache.Low[lowOffset] >> pairShift) & 3;
        var pair1 = (cache.Low[lowOffset + signGroups] >> pairShift) & 3;
        var pair2 = (cache.Low[lowOffset + 2 * signGroups] >> pairShift) & 3;
        return (byte)((((pairSign & 2) << 6) | ((pairSign & 1) << 3))
            | (((pair2 & 2) << 5) | ((pair2 & 1) << 2))
            | (((pair1 & 2) << 4) | ((pair1 & 1) << 1))
            | (((pair0 & 2) << 3) | (pair0 & 1)));
    }

    private static void UpdateHeapWithTies(
        float[] scores,
        int[] indices,
        ref int size,
        int capacity,
        float score,
        int index)
    {
        if (size < capacity)
        {
            scores[size] = score;
            indices[size] = index;
            SiftUp(scores, indices, size++);
        }
        else if (score > scores[0] || (score == scores[0] && index < indices[0]))
        {
            scores[0] = score;
            indices[0] = index;
            SiftDown(scores, indices, 0, size);
        }
    }

    private static int CountAllowedBlocks(ReadOnlySpan<ulong> mask, int nVectors, int nBlocks)
    {
        if (mask.IsEmpty) return nBlocks;
        var count = 0;
        for (var block = 0; block < nBlocks; block++)
        {
            if (BlockHasAllowed(mask, block * Constants.Block, nVectors)) count++;
        }
        return count;
    }

    private static int UpstreamThreadCount() => PlaneThreadCount.Value;
}
