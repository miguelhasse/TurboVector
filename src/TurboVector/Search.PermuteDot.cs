using System.Buffers;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace TurboVector;

public static partial class Search
{
    internal static (float[] Scores, int[] Indices) Score2BitPermuteDot(
        ReadOnlySpan<float> query,
        ReadOnlySpan<byte> blockedCodes,
        ReadOnlySpan<float> centroids,
        ReadOnlySpan<float> vecScales,
        int nByteGroups,
        int nVectors,
        int nBlocks,
        int k,
        ReadOnlySpan<ulong> mask,
        float bias)
    {
        Span<sbyte> levels = stackalloc sbyte[4];
        var weights = ArrayPool<sbyte>.Shared.Rent(query.Length);
        try
        {
            var scale = BuildPermuteOperands(query, centroids, weights, levels);
            if (Dp.IsSupported)
            {
                return ScorePermuteDotArm64(
                    blockedCodes, weights, levels, scale, bias, vecScales,
                    nByteGroups, nVectors, nBlocks, k, mask, 2);
            }
            var scores = CreateScoreBuffer(k);
            var indices = CreateIndexBuffer(k);
            var size = 0;
            var skipped = 0L;
            for (var block = 0; block < nBlocks; block++)
            {
                var baseVector = block * Constants.Block;
                if (baseVector >= nVectors) break;
                if (!BlockHasAllowed(mask, baseVector, nVectors))
                {
                    skipped++;
                    continue;
                }

                var lanes = Math.Min(Constants.Block, nVectors - baseVector);
                for (var lane = 0; lane < lanes; lane++)
                {
                    var v = baseVector + lane;
                    if (!IsAllowed(mask, v)) continue;
                    var sum = 0;
                    for (var group = 0; group < nByteGroups; group++)
                    {
                        var code = blockedCodes[(block * nByteGroups + group) * Constants.Block + lane];
                        var d = group * 4;
                        sum += weights[d] * levels[code >> 6];
                        sum += weights[d + 1] * levels[(code >> 4) & 3];
                        sum += weights[d + 2] * levels[(code >> 2) & 3];
                        sum += weights[d + 3] * levels[code & 3];
                    }
                    var score = MathF.FusedMultiplyAdd(sum, scale, bias) * vecScales[v];
                    UpdateHeap(scores, indices, ref size, k, score, v);
                }
            }
            FinalizeTopK(scores, indices, size);
            if (skipped > 0) Interlocked.Add(ref _blocksSkippedByMask, skipped);
            return (scores, indices);
        }
        finally
        {
            ArrayPool<sbyte>.Shared.Return(weights);
        }
    }

    internal static (float[] Scores, int[] Indices) ScorePermuteDot(
        ReadOnlySpan<float> query,
        ReadOnlySpan<byte> blockedCodes,
        ReadOnlySpan<float> centroids,
        ReadOnlySpan<float> vecScales,
        int nByteGroups,
        int nVectors,
        int nBlocks,
        int k,
        ReadOnlySpan<ulong> mask,
        float bias)
    {
        Span<sbyte> levels = stackalloc sbyte[16];
        var weights = ArrayPool<sbyte>.Shared.Rent(query.Length);
        try
        {
            var combinedScale = BuildPermuteOperands(query, centroids, weights, levels);
            if (Avx2.IsSupported && Fma.IsSupported)
            {
                return ScorePermuteDotAvx2(
                    blockedCodes, weights, levels, combinedScale, bias, vecScales,
                    nByteGroups, nVectors, nBlocks, k, mask);
            }

            if (Dp.IsSupported)
            {
                return ScorePermuteDotArm64(
                    blockedCodes, weights, levels, combinedScale, bias, vecScales,
                    nByteGroups, nVectors, nBlocks, k, mask, 4);
            }

            return ScorePermuteDotScalar(
                blockedCodes, weights, levels, combinedScale, bias, vecScales,
                nByteGroups, nVectors, nBlocks, k, mask);
        }
        finally
        {
            ArrayPool<sbyte>.Shared.Return(weights);
        }
    }

    private static float BuildPermuteOperands(
        ReadOnlySpan<float> query,
        ReadOnlySpan<float> centroids,
        Span<sbyte> weights,
        Span<sbyte> levels)
    {
        var maxCentroid = 0f;
        foreach (var centroid in centroids)
        {
            maxCentroid = MathF.Max(maxCentroid, MathF.Abs(centroid));
        }
        var centroidScale = maxCentroid > 0f ? maxCentroid / 127f : 1f;
        for (var code = 0; code < levels.Length; code++)
        {
            levels[code] = QuantizeSigned(centroids[code] / centroidScale);
        }

        var maxQuery = 0f;
        foreach (var value in query)
        {
            maxQuery = MathF.Max(maxQuery, MathF.Abs(value));
        }

        var queryScale = maxQuery / 127f;
        if (queryScale < 1.17549435e-38f) queryScale = 1f;
        var inverseQueryScale = 1f / queryScale;
        for (var d = 0; d < query.Length; d++)
        {
            weights[d] = QuantizeSigned(query[d] * inverseQueryScale);
        }

        return centroidScale * queryScale;
    }

    private static sbyte QuantizeSigned(float value)
    {
        var rounded = MathF.Round(value, MidpointRounding.AwayFromZero);
        return (sbyte)Math.Clamp(rounded, -127f, 127f);
    }

    private static (float[] Scores, int[] Indices) ScorePermuteDotArm64(
        ReadOnlySpan<byte> blockedCodes,
        ReadOnlySpan<sbyte> weights,
        ReadOnlySpan<sbyte> levels,
        float scale,
        float bias,
        ReadOnlySpan<float> vecScales,
        int nByteGroups,
        int nVectors,
        int nBlocks,
        int k,
        ReadOnlySpan<ulong> mask,
        int bits)
    {
        var heapScores = CreateScoreBuffer(k);
        var heapIndices = CreateIndexBuffer(k);
        var heapSize = 0;
        var skipped = 0L;
        Span<sbyte> levelBuffer = stackalloc sbyte[16];
        levelBuffer.Clear();
        levels.CopyTo(levelBuffer);
        var table = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(levelBuffer)).AsByte();
        var codeMask = Vector128.Create((byte)(bits == 2 ? 3 : 15));
        var vectorScale = Vector128.Create(scale);
        var vectorBias = Vector128.Create(bias);
        Span<Vector128<int>> sums = stackalloc Vector128<int>[8];
        Span<float> blockScores = stackalloc float[Constants.Block];
        ref var codeRef = ref MemoryMarshal.GetReference(blockedCodes);
        ref var scoreRef = ref MemoryMarshal.GetReference(blockScores);
        var groupStep = bits == 2 ? 1 : 2;
        for (var block = 0; block < nBlocks; block++)
        {
            var baseVector = block * Constants.Block;
            if (baseVector >= nVectors) break;
            if (!BlockHasAllowed(mask, baseVector, nVectors))
            {
                skipped++;
                continue;
            }
            sums.Clear();
            for (var group = 0; group < nByteGroups; group += groupStep)
            {
                var weightOffset = bits == 2 ? group * 4 : group * 2;
                var packedWeights = (uint)(byte)weights[weightOffset]
                    | (uint)(byte)weights[weightOffset + 1] << 8
                    | (uint)(byte)weights[weightOffset + 2] << 16
                    | (uint)(byte)weights[weightOffset + 3] << 24;
                var queryWeights = Vector128.Create(packedWeights).AsSByte();
                for (var half = 0; half < 2; half++)
                {
                    var offset = (nuint)((block * nByteGroups + group) * Constants.Block + half * 16);
                    var code = Vector128.LoadUnsafe(ref codeRef, offset);
                    var first = Vector128<byte>.Zero;
                    var second = Vector128<byte>.Zero;
                    var third = Vector128<byte>.Zero;
                    var fourth = Vector128<byte>.Zero;
                    if (bits == 2)
                    {
                        first = Vector128.Shuffle(table, Vector128.BitwiseAnd(Vector128.ShiftRightLogical(code.AsUInt16(), 6).AsByte(), codeMask));
                        second = Vector128.Shuffle(table, Vector128.BitwiseAnd(Vector128.ShiftRightLogical(code.AsUInt16(), 4).AsByte(), codeMask));
                        third = Vector128.Shuffle(table, Vector128.BitwiseAnd(Vector128.ShiftRightLogical(code.AsUInt16(), 2).AsByte(), codeMask));
                        fourth = Vector128.Shuffle(table, Vector128.BitwiseAnd(code, codeMask));
                    }
                    else
                    {
                        var next = Vector128.LoadUnsafe(ref codeRef, offset + Constants.Block);
                        first = Vector128.Shuffle(table, Vector128.BitwiseAnd(Vector128.ShiftRightLogical(code.AsUInt16(), 4).AsByte(), codeMask));
                        second = Vector128.Shuffle(table, Vector128.BitwiseAnd(code, codeMask));
                        third = Vector128.Shuffle(table, Vector128.BitwiseAnd(Vector128.ShiftRightLogical(next.AsUInt16(), 4).AsByte(), codeMask));
                        fourth = Vector128.Shuffle(table, Vector128.BitwiseAnd(next, codeMask));
                    }

                    var firstLow = AdvSimd.Arm64.ZipLow(first, second).AsUInt16();
                    var secondLow = AdvSimd.Arm64.ZipLow(third, fourth).AsUInt16();
                    var firstHigh = AdvSimd.Arm64.ZipHigh(first, second).AsUInt16();
                    var secondHigh = AdvSimd.Arm64.ZipHigh(third, fourth).AsUInt16();
                    var slot = half * 4;
                    sums[slot] = Dp.DotProduct(sums[slot], AdvSimd.Arm64.ZipLow(firstLow, secondLow).AsSByte(), queryWeights);
                    sums[slot + 1] = Dp.DotProduct(sums[slot + 1], AdvSimd.Arm64.ZipHigh(firstLow, secondLow).AsSByte(), queryWeights);
                    sums[slot + 2] = Dp.DotProduct(sums[slot + 2], AdvSimd.Arm64.ZipLow(firstHigh, secondHigh).AsSByte(), queryWeights);
                    sums[slot + 3] = Dp.DotProduct(sums[slot + 3], AdvSimd.Arm64.ZipHigh(firstHigh, secondHigh).AsSByte(), queryWeights);
                }
            }

            for (var slot = 0; slot < sums.Length; slot++)
            {
                AdvSimd.FusedMultiplyAdd(vectorBias, vectorScale, Vector128.ConvertToSingle(sums[slot]))
                    .StoreUnsafe(ref scoreRef, (nuint)(slot * 4));
            }
            var lanes = Math.Min(Constants.Block, nVectors - baseVector);
            for (var lane = 0; lane < lanes; lane++)
            {
                var index = baseVector + lane;
                if (!IsAllowed(mask, index)) continue;
                UpdateHeap(heapScores, heapIndices, ref heapSize, k, blockScores[lane] * vecScales[index], index);
            }
        }
        FinalizeTopK(heapScores, heapIndices, heapSize);
        if (skipped > 0) Interlocked.Add(ref _blocksSkippedByMask, skipped);
        return (heapScores, heapIndices);
    }

    private static (float[] Scores, int[] Indices) ScorePermuteDotScalar(
        ReadOnlySpan<byte> blockedCodes,
        ReadOnlySpan<sbyte> weights,
        ReadOnlySpan<sbyte> levels,
        float scale,
        float bias,
        ReadOnlySpan<float> vecScales,
        int nByteGroups,
        int nVectors,
        int nBlocks,
        int k,
        ReadOnlySpan<ulong> mask)
    {
        var heapScores = CreateScoreBuffer(k);
        var heapIndices = CreateIndexBuffer(k);
        var heapSize = 0;
        var skipped = 0L;
        for (var block = 0; block < nBlocks; block++)
        {
            var baseVector = block * Constants.Block;
            if (baseVector >= nVectors) break;
            if (!BlockHasAllowed(mask, baseVector, nVectors))
            {
                skipped++;
                continue;
            }

            var lanes = Math.Min(Constants.Block, nVectors - baseVector);
            for (var lane = 0; lane < lanes; lane++)
            {
                var index = baseVector + lane;
                if (!IsAllowed(mask, index)) continue;
                var sum = 0;
                for (var group = 0; group < nByteGroups; group++)
                {
                    var code = blockedCodes[(block * nByteGroups + group) * Constants.Block + lane];
                    sum += weights[group * 2] * levels[code >> 4];
                    sum += weights[group * 2 + 1] * levels[code & 15];
                }

                var score = MathF.FusedMultiplyAdd(sum, scale, bias) * vecScales[index];
                UpdateHeap(heapScores, heapIndices, ref heapSize, k, score, index);
            }
        }

        FinalizeTopK(heapScores, heapIndices, heapSize);
        if (skipped > 0) Interlocked.Add(ref _blocksSkippedByMask, skipped);
        return (heapScores, heapIndices);
    }

    private static (float[] Scores, int[] Indices) ScorePermuteDotAvx2(
        ReadOnlySpan<byte> blockedCodes,
        ReadOnlySpan<sbyte> weights,
        ReadOnlySpan<sbyte> levels,
        float scale,
        float bias,
        ReadOnlySpan<float> vecScales,
        int nByteGroups,
        int nVectors,
        int nBlocks,
        int k,
        ReadOnlySpan<ulong> mask)
    {
        var heapScores = CreateScoreBuffer(k);
        var heapIndices = CreateIndexBuffer(k);
        var heapSize = 0;
        var skipped = 0L;
        var nibbleMask = Vector256.Create((byte)15);
        var vectorScale = Vector256.Create(scale);
        var vectorBias = Vector256.Create(bias);
        var table = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(levels)).AsByte();
        var levelTable = Vector256.Create(table, table);
        Span<float> blockScores = stackalloc float[Constants.Block];
        ref var codeRef = ref MemoryMarshal.GetReference(blockedCodes);
        ref var scoreRef = ref MemoryMarshal.GetReference(blockScores);
        for (var block = 0; block < nBlocks; block++)
        {
            var baseVector = block * Constants.Block;
            if (baseVector >= nVectors) break;
            if (!BlockHasAllowed(mask, baseVector, nVectors))
            {
                skipped++;
                continue;
            }

            var sum0 = Vector256<int>.Zero;
            var sum1 = Vector256<int>.Zero;
            var sum2 = Vector256<int>.Zero;
            var sum3 = Vector256<int>.Zero;
            for (var group = 0; group < nByteGroups; group++)
            {
                var codes = Vector256.LoadUnsafe(ref codeRef, (nuint)((block * nByteGroups + group) * Constants.Block));
                var high = Avx2.And(Avx2.ShiftRightLogical(codes.AsUInt16(), 4).AsByte(), nibbleMask);
                var low = Avx2.And(codes, nibbleMask);
                var first = Avx2.Shuffle(levelTable, high).AsSByte();
                var second = Avx2.Shuffle(levelTable, low).AsSByte();
                var pair0 = Avx2.UnpackLow(first, second);
                var pair1 = Avx2.UnpackHigh(first, second);
                var pairBits = (ushort)((byte)weights[group * 2] | ((ushort)(byte)weights[group * 2 + 1] << 8));
                var pairWeights = Vector256.Create(pairBits).AsSByte();
                var products0 = Avx2.MultiplyAddAdjacent(Avx2.Abs(pair0), Avx2.Sign(pairWeights, pair0));
                var products1 = Avx2.MultiplyAddAdjacent(Avx2.Abs(pair1), Avx2.Sign(pairWeights, pair1));
                sum0 = Avx2.Add(sum0, Avx2.ConvertToVector256Int32(products0.GetLower()));
                sum1 = Avx2.Add(sum1, Avx2.ConvertToVector256Int32(products1.GetLower()));
                sum2 = Avx2.Add(sum2, Avx2.ConvertToVector256Int32(products0.GetUpper()));
                sum3 = Avx2.Add(sum3, Avx2.ConvertToVector256Int32(products1.GetUpper()));
            }

            var f0 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(sum0), vectorScale, vectorBias);
            var f1 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(sum1), vectorScale, vectorBias);
            var f2 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(sum2), vectorScale, vectorBias);
            var f3 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(sum3), vectorScale, vectorBias);
            f0.StoreUnsafe(ref scoreRef);
            f1.StoreUnsafe(ref scoreRef, 8);
            f2.StoreUnsafe(ref scoreRef, 16);
            f3.StoreUnsafe(ref scoreRef, 24);
            var lanes = Math.Min(Constants.Block, nVectors - baseVector);
            for (var lane = 0; lane < lanes; lane++)
            {
                var index = baseVector + lane;
                if (!IsAllowed(mask, index)) continue;
                var score = blockScores[lane] * vecScales[index];
                UpdateHeap(heapScores, heapIndices, ref heapSize, k, score, index);
            }
        }

        FinalizeTopK(heapScores, heapIndices, heapSize);
        if (skipped > 0) Interlocked.Add(ref _blocksSkippedByMask, skipped);
        return (heapScores, heapIndices);
    }
}
