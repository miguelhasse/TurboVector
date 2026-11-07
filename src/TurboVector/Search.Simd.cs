using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace TurboVector;

public static partial class Search
{
    private static (float[] Scores, int[] Indices) ScoreNibbleAvx2(
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
        bool vectorMajor,
        float[]? allScores = null)
    {
        var heapScores = allScores is null ? CreateScoreBuffer(k) : [];
        var heapIndices = allScores is null ? CreateIndexBuffer(k) : [];
        var heapSize = 0;
        var skipped = 0L;
        var nibbleMask = Vector256.Create((byte)0x0F);
        var vectorScale = Vector256.Create(scale);
        var vectorBias = Vector256.Create(bias);
        Span<float> blockScores = stackalloc float[Constants.Block];
        ref var codeRef = ref MemoryMarshal.GetReference(blockedCodes);
        ref var lutRef = ref MemoryMarshal.GetArrayDataReference(lut);
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

            var f0 = vectorBias;
            var f1 = vectorBias;
            var f2 = vectorBias;
            var f3 = vectorBias;
            var sum0 = Vector256<int>.Zero;
            var sum1 = Vector256<int>.Zero;
            var sum2 = Vector256<int>.Zero;
            var sum3 = Vector256<int>.Zero;
            for (var start = 0; start < nByteGroups; start += 256)
            {
                var acc0 = Vector256<ushort>.Zero;
                var acc1 = Vector256<ushort>.Zero;
                var end = Math.Min(start + 256, nByteGroups);
                for (var group = start; group < end; group++)
                {
                    var codes = Vector256.LoadUnsafe(ref codeRef, (nuint)((block * nByteGroups + group) * Constants.Block));
                    var high = Avx2.And(Avx2.ShiftRightLogical(codes.AsUInt16(), 4).AsByte(), nibbleMask);
                    var low = Avx2.And(codes, nibbleMask);
                    var first = Vector128.LoadUnsafe(ref lutRef, (nuint)(group * 32));
                    var second = Vector128.LoadUnsafe(ref lutRef, (nuint)(group * 32 + 16));
                    var values = Avx2.Add(
                        Avx2.Shuffle(Vector256.Create(first, first), high),
                        Avx2.Shuffle(Vector256.Create(second, second), low));
                    acc0 = Avx2.Add(acc0, Avx2.ConvertToVector256Int16(values.GetLower()).AsUInt16());
                    acc1 = Avx2.Add(acc1, Avx2.ConvertToVector256Int16(values.GetUpper()).AsUInt16());
                }

                var part0 = Avx2.ConvertToVector256Int32(acc0.GetLower());
                var part1 = Avx2.ConvertToVector256Int32(acc0.GetUpper());
                var part2 = Avx2.ConvertToVector256Int32(acc1.GetLower());
                var part3 = Avx2.ConvertToVector256Int32(acc1.GetUpper());
                if (vectorMajor)
                {
                    sum0 = Avx2.Add(sum0, part0);
                    sum1 = Avx2.Add(sum1, part1);
                    sum2 = Avx2.Add(sum2, part2);
                    sum3 = Avx2.Add(sum3, part3);
                }
                else
                {
                    f0 = Fma.MultiplyAdd(vectorScale, Avx.ConvertToVector256Single(part0), f0);
                    f1 = Fma.MultiplyAdd(vectorScale, Avx.ConvertToVector256Single(part1), f1);
                    f2 = Fma.MultiplyAdd(vectorScale, Avx.ConvertToVector256Single(part2), f2);
                    f3 = Fma.MultiplyAdd(vectorScale, Avx.ConvertToVector256Single(part3), f3);
                }
            }

            if (vectorMajor)
            {
                f0 = Avx.Add(Avx.Multiply(Avx.ConvertToVector256Single(sum0), vectorScale), vectorBias);
                f1 = Avx.Add(Avx.Multiply(Avx.ConvertToVector256Single(sum1), vectorScale), vectorBias);
                f2 = Avx.Add(Avx.Multiply(Avx.ConvertToVector256Single(sum2), vectorScale), vectorBias);
                f3 = Avx.Add(Avx.Multiply(Avx.ConvertToVector256Single(sum3), vectorScale), vectorBias);
            }

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
                if (allScores is null) UpdateHeap(heapScores, heapIndices, ref heapSize, k, score, index);
                else allScores[index] = score;
            }
        }

        FinalizeTopK(heapScores, heapIndices, heapSize);
        if (skipped > 0) Interlocked.Add(ref _blocksSkippedByMask, skipped);
        return (heapScores, heapIndices);
    }

    private static (float[] Scores, int[] Indices) ScoreNibbleArm64(
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
        float[]? allScores = null)
    {
        var heapScores = allScores is null ? CreateScoreBuffer(k) : [];
        var heapIndices = allScores is null ? CreateIndexBuffer(k) : [];
        var heapSize = 0;
        var skipped = 0L;
        var nibbleMask = Vector128.Create((byte)0x0F);
        var vectorScale = Vector128.Create(scale);
        var vectorBias = Vector128.Create(bias);
        Span<Vector128<float>> accumulators = stackalloc Vector128<float>[8];
        Span<Vector128<ushort>> sums = stackalloc Vector128<ushort>[4];
        Span<float> blockScores = stackalloc float[Constants.Block];
        ref var codeRef = ref MemoryMarshal.GetReference(blockedCodes);
        ref var lutRef = ref MemoryMarshal.GetArrayDataReference(lut);
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

            accumulators.Fill(vectorBias);
            for (var start = 0; start < nByteGroups; start += 256)
            {
                sums.Clear();
                var end = Math.Min(start + 256, nByteGroups);
                for (var group = start; group < end; group++)
                {
                    var offset = (nuint)((block * nByteGroups + group) * Constants.Block);
                    var codes0 = Vector128.LoadUnsafe(ref codeRef, offset);
                    var codes1 = Vector128.LoadUnsafe(ref codeRef, offset + 16);
                    var first = Vector128.LoadUnsafe(ref lutRef, (nuint)(group * 32));
                    var second = Vector128.LoadUnsafe(ref lutRef, (nuint)(group * 32 + 16));
                    var high0 = Vector128.BitwiseAnd(Vector128.ShiftRightLogical(codes0.AsUInt16(), 4).AsByte(), nibbleMask);
                    var high1 = Vector128.BitwiseAnd(Vector128.ShiftRightLogical(codes1.AsUInt16(), 4).AsByte(), nibbleMask);
                    var values0 = Vector128.Add(
                        Vector128.Shuffle(first, high0),
                        Vector128.Shuffle(second, Vector128.BitwiseAnd(codes0, nibbleMask)));
                    var values1 = Vector128.Add(
                        Vector128.Shuffle(first, high1),
                        Vector128.Shuffle(second, Vector128.BitwiseAnd(codes1, nibbleMask)));
                    sums[0] = Vector128.Add(sums[0], Vector128.WidenLower(values0));
                    sums[1] = Vector128.Add(sums[1], Vector128.WidenUpper(values0));
                    sums[2] = Vector128.Add(sums[2], Vector128.WidenLower(values1));
                    sums[3] = Vector128.Add(sums[3], Vector128.WidenUpper(values1));
                }

                for (var i = 0; i < sums.Length; i++)
                {
                    var lo = Vector128.ConvertToSingle(Vector128.WidenLower(sums[i]).AsInt32());
                    var hi = Vector128.ConvertToSingle(Vector128.WidenUpper(sums[i]).AsInt32());
                    accumulators[i * 2] = AdvSimd.FusedMultiplyAdd(accumulators[i * 2], vectorScale, lo);
                    accumulators[i * 2 + 1] = AdvSimd.FusedMultiplyAdd(accumulators[i * 2 + 1], vectorScale, hi);
                }
            }

            for (var i = 0; i < accumulators.Length; i++)
            {
                accumulators[i].StoreUnsafe(ref scoreRef, (nuint)(i * 4));
            }

            var lanes = Math.Min(Constants.Block, nVectors - baseVector);
            for (var lane = 0; lane < lanes; lane++)
            {
                var index = baseVector + lane;
                if (!IsAllowed(mask, index)) continue;
                var score = blockScores[lane] * vecScales[index];
                if (allScores is null) UpdateHeap(heapScores, heapIndices, ref heapSize, k, score, index);
                else allScores[index] = score;
            }
        }

        FinalizeTopK(heapScores, heapIndices, heapSize);
        if (skipped > 0) Interlocked.Add(ref _blocksSkippedByMask, skipped);
        return (heapScores, heapIndices);
    }
}
