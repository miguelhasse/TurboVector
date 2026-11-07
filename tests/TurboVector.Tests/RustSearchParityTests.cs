using System.Buffers.Binary;
using System.Globalization;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace TurboVector.Tests;

[CollectionDefinition(nameof(SearchCounterCollection), DisableParallelization = true)]
public sealed class SearchCounterCollection { }

[Collection(nameof(SearchCounterCollection))]
public class RustSearchParityTests
{
    private const string TwoBitBatchExpectedIds = "32,102,78,84,124,82,24,0,32,20,114,16,6,2,106,98,96,126,20,40";
    private const string TwoBitBatchExpectedScores = "40d8c6db,40b1c6d2,40abf40c,40a6aa93,408f71d8,40e23391,40d01d3a,40ae6b3a,407a9920,406e9140,40bd194c,40a2cc7a,409ac525,40999ae6,40960d6b,40b69633,40a4ca90,409eeae6,409932e2,408a051e";

    [Theory]
    [InlineData(2, "0cff3c5d1defcf1a", "b68b16120fd9c2d2", "124,59,17,19,32", "40ce454f,40cc18bc,40c3557a,40b9c3f3,40b11e01")]
    [InlineData(3, "eb61e24b45b7a8f1", "5093fd329898a558", "124,59,17,19,78", "40dec155,40bfc4f1,40be0491,409cd1e9,40946347")]
    [InlineData(4, "6d0813f884a3bc33", "9a581d17842ffbad", "59,17,124,19,32", "40c78316,40be47fb,40b2a197,40a9a940,409bac71")]
    public void ClassicScoringMatchesRust(
        int bits, string codeHash, string scaleHash, string expectedIds, string expectedScores)
    {
        Assert.SkipWhen(!(Avx2.IsSupported && Fma.IsSupported) && !AdvSimd.Arm64.IsSupported,
            "The Rust fixture uses the SIMD classic scorer.");
        const int dim = 64;
        const int count = 128;
        var index = CreateIndex(dim, count, bits);
        Assert.Equal(codeHash, Hash(index.PackedCodes));
        Assert.Equal(scaleHash, HashScales(index.Scales));
        var (blocked, blocks) = Pack.Repack(index.PackedCodes, count, bits, dim);
        var query = RotatedQuery(index, Rows(1, dim, 0xBADF00DUL), out var bias);
        var centroids = Codebook.Compute(bits, dim).Centroids;
        var (scores, ids) = Search.ScoreBlockedQuery(query, blocked, centroids, index.Scales,
            bits, dim / (8 / bits), count, blocks, 5, [], bias, false);
        AssertResult(ids, scores, expectedIds, expectedScores);
    }

    [Theory]
    [InlineData(2, 1024, "b99382f285ba3d0e", "b72571b6f061ed08", "57,0,54,7,9", "41ee8f8a,41b9294d,41a20b0a,419ff371,419e3eb5")]
    [InlineData(2, 1032, "6c6abcd484039b3a", "c39d16acd095964c", "46,49,56,0,5", "41f4d537,41db0a2c,41d2128b,41b04cf9,419241bc")]
    [InlineData(3, 1024, "7d6e35fcb2bde8e0", "fd16afbfa80de79d", "57,0,7,39,54", "41f97b82,41bf8cd7,41a262d3,418df617,417daea3")]
    [InlineData(3, 1032, "517c2f91442b7a3d", "1c43800d7578b6a6", "49,56,0,46,5", "41f01825,41c5a005,41c44208,41bafdf4,41b33d2f")]
    [InlineData(4, 1024, "5357ff21409419d5", "7030784151ec4f36", "57,0,7,53,64", "420150f9,41c30cf2,419ff149,4184549e,4180fb0a")]
    [InlineData(4, 1032, "d27f01447349779c", "79346304ccbe342d", "0,46,49,56,5", "41dd728d,41cd05e5,41cbbc83,41b75d38,419250fa")]
    public void ClassicScoringMatchesRustAcrossFlushBoundaries(
        int bits, int dim, string codeHash, string scaleHash, string expectedIds, string expectedScores)
    {
        Assert.SkipWhen(!(Avx2.IsSupported && Fma.IsSupported) && !AdvSimd.Arm64.IsSupported,
            "The Rust fixture uses the SIMD classic scorer.");
        const int count = 65;
        var index = CreateIndex(dim, count, bits);
        Assert.Equal(codeHash, Hash(index.PackedCodes));
        Assert.Equal(scaleHash, HashScales(index.Scales));
        var (blocked, blocks) = Pack.Repack(index.PackedCodes, count, bits, dim);
        var query = RotatedQuery(index, Rows(1, dim, 0xBADF00DUL), out var bias);
        var centroids = Codebook.Compute(bits, dim).Centroids;
        var (scores, ids) = Search.ScoreBlockedQuery(query, blocked, centroids, index.Scales,
            bits, dim / (8 / bits), count, blocks, 5, [], bias, false);
        AssertResult(ids, scores, expectedIds, expectedScores);
    }

    [Theory]
    [InlineData(2, "b84f23ea551d5471", "cc3d07f3cfbdee38", "32,102,78,84,124,82,24,0,32,20", "40d6ed7c,40b38132,40aa9d82,40a4f552,408eade2,40e16a64,40d27a9e,40aeb108,4075ed83,4070e599")]
    [InlineData(3, "16e846025fa00b02", "f3c9a417a7b120b0", "124,32,102,118,100,24,82,0,56,28", "40bebba9,40b7c8f7,408d443f,4085b1e9,408301b8,40de90e9,40b9a2a8,40b417be,40863066,407f1d32")]
    [InlineData(4, "c40ddedc4cfe2ead", "5b94ba551effe296", "124,32,78,102,116,24,82,0,118,56", "40c2d19d,409d30c4,40741a5b,40729452,406f6dfb,40f726fc,40be2acf,40a745e8,40653786,4062ae81")]
    public void VectorMajorScoringMatchesRust(
        int bits, string codeHash, string scaleHash, string expectedIds, string expectedScores)
    {
        Assert.SkipWhen(!(Avx2.IsSupported && Fma.IsSupported) && (bits != 4 || !Dp.IsSupported),
            "These fixtures exercise vector-major arithmetic.");
        const int dim = 64;
        const int count = 129;
        var index = CreateIndex(dim, count, bits, calibrated: true);
        Assert.Equal(codeHash, Hash(index.PackedCodes));
        Assert.Equal(scaleHash, HashScales(index.Scales));
        var (blocked, blocks) = Pack.Repack(index.PackedCodes, count, bits, dim);
        var centroids = Codebook.Compute(bits, dim).Centroids;
        var mask = new ulong[(count + 63) / 64];
        for (var i = 0; i < count; i += 2) mask[i / 64] |= 1UL << (i % 64);
        var queries = Rows(2, dim, 0xBADF00DUL);
        var allScores = new List<float>();
        var allIds = new List<int>();
        for (var qi = 0; qi < 2; qi++)
        {
            var query = RotatedQuery(index, queries.AsSpan(qi * dim, dim), out var bias);
            var (scores, ids) = Search.ScoreBlockedQuery(query, blocked, centroids, index.Scales,
                bits, dim / (8 / bits), count, blocks, 5, mask, bias, true);
            allScores.AddRange(scores);
            allIds.AddRange(ids);
        }
        AssertResult(allIds, allScores, expectedIds, expectedScores);
    }

    [Theory]
    [InlineData(2, "normal", "26820,22118,30944,20650,22732", "413ba193,412fccb8,41295433,41264bbd,41263cb5")]
    [InlineData(2, "zero", "8,10,12,14,16", "00000000,00000000,00000000,00000000,00000000")]
    [InlineData(2, "duplicates", "8,12,16,20,24", "41bbcf3e,41bbcf3e,41bbcf3e,41bbcf3e,41bbcf3e")]
    [InlineData(4, "normal", "26820,14738,8070,4076,12572", "413688c4,4118ea30,41189fcb,41177bf8,41153d0b")]
    [InlineData(4, "zero", "8,10,12,14,16", "00000000,00000000,00000000,00000000,00000000")]
    [InlineData(4, "duplicates", "8,12,16,20,24", "41bb41c2,41bb41c2,41bb41c2,41bb41c2,41bb41c2")]
    public void StagedSelectionMatchesRustIncludingTies(
        int bits, string mode, string expectedIds, string expectedScores)
    {
        const int dim = 64;
        const int count = 32768;
        var vectors = Rows(count, dim, 0xC0FFEEUL);
        var originalQuery = Rows(1, dim, 0xBADF00DUL);
        if (mode == "zero")
        {
            Array.Clear(originalQuery);
        }
        else if (mode == "duplicates")
        {
            var first = vectors.AsSpan(0, dim).ToArray();
            for (var i = 0; i < count; i += 4) first.CopyTo(vectors, i * dim);
            first.CopyTo(originalQuery, 0);
        }

        var index = new TurboQuantIndex(dim, bits);
        index.Add(vectors);
        var (blocked, blocks) = Pack.Repack(index.PackedCodes, count, bits, dim);
        var centroids = Codebook.Compute(bits, dim).Centroids;
        var cache = Search.BuildPlanesCache(blocked, centroids, [], [], bits, dim, count, blocks);
        var mask = new ulong[count / 64];
        for (var i = 0; i < count; i += 2) mask[i / 64] |= 1UL << (i % 64);
        var query = RotatedQuery(index, originalQuery, out var bias);
        var (scores, ids) = Search.ScorePlanesQuery(query, centroids, index.Scales, cache,
            bits, count, blocks, 5, mask, bias, [], [], 1, 1, false);
        AssertResult(ids, scores, expectedIds, expectedScores);
    }

    [Theory]
    [InlineData(2, "26820,22118,30944,20650,22732", "413ba194,412fccb9,41295435,41264bbe,41263cb6")]
    [InlineData(4, "26820,14738,8070,4076,12572", "413688c4,4118ea30,41189fcb,41177bf8,41153d0b")]
    public void ArmStagedScoringMatchesRust(int bits, string expectedIds, string expectedScores)
    {
        const int dim = 64;
        const int count = 32768;
        var index = CreateIndex(dim, count, bits);
        var (blocked, blocks) = Pack.Repack(index.PackedCodes, count, bits, dim);
        var centroids = Codebook.Compute(bits, dim).Centroids;
        var cache = Search.BuildPlanesCache(blocked, centroids, [], [], bits, dim, count, blocks);
        var mask = new ulong[count / 64];
        for (var i = 0; i < count; i += 2) mask[i / 64] |= 1UL << (i % 64);
        var query = RotatedQuery(index, Rows(1, dim, 0xBADF00DUL), out var bias);
        var (scores, ids) = Search.ScorePlanesQuery(query, centroids, index.Scales, cache,
            bits, count, blocks, 5, mask, bias, [], [], 1, 1, true);
        AssertResult(ids, scores, expectedIds, expectedScores);
    }

    [Theory]
    [InlineData(2, false, 1, "321,322,323,324,325,4,5,6,7,8", "41afc909")]
    [InlineData(2, false, 4, "324,325,326,327,328,4,5,6,7,8", "41afc909")]
    [InlineData(2, true, 1, "321,322,323,324,325,4,5,6,7,8", "41afc90a")]
    [InlineData(2, true, 4, "324,325,326,327,328,4,5,6,7,8", "41afc90a")]
    [InlineData(4, false, 1, "322,323,324,325,326,4,5,6,7,8", "41ae2e4b")]
    [InlineData(4, false, 4, "324,325,326,327,328,4,5,6,7,8", "41ae2e4b")]
    [InlineData(4, true, 1, "322,323,324,325,326,4,5,6,7,8", "41ae2e4b")]
    [InlineData(4, true, 4, "324,325,326,327,328,4,5,6,7,8", "41ae2e4b")]
    public void StagedBatchRetriesMatchRust(int bits, bool arm64, int threads, string expectedIds, string scoreBits)
    {
        const int dim = 64;
        const int count = 32768;
        var (index, queries) = CreateSampleSkewIndex(bits);
        var (blocked, blocks) = Pack.Repack(index.PackedCodes, count, bits, dim);
        var centroids = Codebook.Compute(bits, dim).Centroids;
        var cache = Search.BuildPlanesCache(blocked, centroids, [], [], bits, dim, count, blocks);
        var shortQueries = new bool[2];
        var rotated = new float[2][];
        var biases = new float[2];
        var scores = new float[2][];
        var indices = new int[2][];
        for (var qi = 0; qi < shortQueries.Length; qi++)
        {
            rotated[qi] = RotatedQuery(index, queries.AsSpan(qi * dim, dim), out biases[qi]);
            (scores[qi], indices[qi]) = Search.ScorePlanesQuery(
                rotated[qi], centroids, index.Scales, cache, bits, count, blocks, 5, [],
                biases[qi], [], [], 2, threads, arm64, shortQueries, qi);
        }

        Assert.Equal([true, false], shortQueries);
        (scores[0], indices[0]) = Search.ScorePlanesQuery(
            rotated[0], centroids, index.Scales, cache, bits, count, blocks, 5, [],
            biases[0], [], [], 2, threads, arm64, shortQueries, 0, 1);
        AssertResult(indices.SelectMany(static value => value), scores.SelectMany(static value => value),
            expectedIds, string.Join(',', Enumerable.Repeat(scoreBits, 10)));
    }

    [Theory]
    [InlineData(2, false, 1, "324,325,326,327,328,4,5,6,7,8", "41ade0e5", "41ae25f4")]
    [InlineData(2, false, 2, "321,322,323,324,325,321,322,323,324,325,4,5,6,7,8", "41ade0e5", "41ae25f4")]
    [InlineData(2, true, 1, "324,325,326,327,328,4,5,6,7,8", "41ade0e6", "41ae25f5")]
    [InlineData(2, true, 2, "321,322,323,324,325,321,322,323,324,325,4,5,6,7,8", "41ade0e6", "41ae25f5")]
    [InlineData(4, false, 1, "324,325,326,327,328,4,5,6,7,8", "41ae3fd7", "41ae4249")]
    [InlineData(4, false, 2, "322,323,324,325,326,322,323,324,325,326,4,5,6,7,8", "41ae3fd7", "41ae4249")]
    [InlineData(4, true, 1, "324,325,326,327,328,4,5,6,7,8", "41ae3fd7", "41ae4249")]
    [InlineData(4, true, 2, "322,323,324,325,326,322,323,324,325,326,4,5,6,7,8", "41ae3fd7", "41ae4249")]
    public void CalibratedStagedBatchRetriesMatchRust(
        int bits, bool arm64, int shortCount, string expectedIds, string shortScore, string otherScore)
    {
        const int dim = 64;
        const int count = 32768;
        var queryCount = shortCount + 1;
        var (index, queries) = CreateSampleSkewIndex(
            bits, calibrated: true, queryCount: queryCount, shortCount: shortCount);
        Assert.Equal(bits == 2 ? "0eb82f257d12c325" : "8f46ee3e9af57725", Hash(index.PackedCodes));
        Assert.Equal(bits == 2 ? "43126fc238ee3b25" : "051a795188611f25", HashScales(index.Scales));
        var (blocked, blocks) = Pack.Repack(index.PackedCodes, count, bits, dim);
        var centroids = Codebook.Compute(bits, dim).Centroids;
        var cache = Search.BuildPlanesCache(blocked, centroids, index.TqPlusShift, index.TqPlusScale,
            bits, dim, count, blocks);
        var shortQueries = new bool[queryCount];
        var rotated = new float[queryCount][];
        var biases = new float[queryCount];
        var scores = new float[queryCount][];
        var indices = new int[queryCount][];
        for (var qi = 0; qi < queryCount; qi++)
        {
            rotated[qi] = RotatedQuery(index, queries.AsSpan(qi * dim, dim), out biases[qi]);
            (scores[qi], indices[qi]) = Search.ScorePlanesQuery(
                rotated[qi], centroids, index.Scales, cache, bits, count, blocks, 5, [],
                biases[qi], index.TqPlusShift, index.TqPlusScale, queryCount, 4, arm64, shortQueries, qi);
        }

        Assert.Equal(Enumerable.Range(0, queryCount).Select(qi => qi < shortCount), shortQueries);
        for (var qi = 0; qi < shortCount; qi++)
        {
            (scores[qi], indices[qi]) = Search.ScorePlanesQuery(
                rotated[qi], centroids, index.Scales, cache, bits, count, blocks, 5, [],
                biases[qi], index.TqPlusShift, index.TqPlusScale, queryCount, 4, arm64, shortQueries, qi, shortCount);
        }
        var expectedScores = Enumerable.Repeat(shortScore, shortCount * 5).Concat(Enumerable.Repeat(otherScore, 5));
        AssertResult(indices.SelectMany(static value => value), scores.SelectMany(static value => value),
            expectedIds, string.Join(',', expectedScores));
    }

    [Theory]
    [InlineData(2, "41afc90a")]
    [InlineData(4, "41ae2e4b")]
    public void ArmIndexCoordinatesStagedBatchRetries(int bits, string scoreBits)
    {
        Assert.SkipWhen(!AdvSimd.Arm64.IsSupported || Environment.GetEnvironmentVariable("RAYON_NUM_THREADS") != "4",
            "This Rust fixture requires ARM64 and four upstream search partitions.");
        Assert.SkipWhen(!Search.UsePlanes(bits, 64 / (8 / bits), 32768),
            "This Rust fixture requires staged search to be enabled.");
        var (index, queries) = CreateSampleSkewIndex(bits);
        var results = index.Search(queries, 5);
        AssertResult(results.Indices.Select(static value => checked((int)value)), results.Scores,
            "324,325,326,327,328,4,5,6,7,8", string.Join(',', Enumerable.Repeat(scoreBits, 10)));
    }

    [Theory]
    [InlineData(2, false, 32768, 2)]
    [InlineData(2, true, 32768, 2)]
    [InlineData(4, false, 32768, 2)]
    [InlineData(4, true, 32768, 2)]
    [InlineData(2, false, 32769, 1)]
    [InlineData(2, true, 32769, 1)]
    [InlineData(4, false, 32769, 1)]
    [InlineData(4, true, 32769, 1)]
    public void StagedSearchIgnoresExtraBlockedCapacity(int bits, bool arm64, int count, int queryCount)
    {
        const int dim = 64;
        const int extraBlocks = 128;
        var index = CreateIndex(dim, count, bits);
        var (blocked, blocks) = Pack.Repack(index.PackedCodes, count, bits, dim);
        var groups = dim / (8 / bits);
        var padded = new byte[blocked.Length + extraBlocks * groups * Constants.Block];
        blocked.CopyTo(padded, 0);
        var centroids = Codebook.Compute(bits, dim).Centroids;
        var cache = Search.BuildPlanesCache(blocked, centroids, [], [], bits, dim, count, blocks);
        var paddedCache = Search.BuildPlanesCache(padded, centroids, [], [], bits, dim, count, blocks + extraBlocks);
        var query = RotatedQuery(index, Rows(1, dim, 0xBADF00DUL), out var bias);

        var expected = Search.ScorePlanesQuery(query, centroids, index.Scales, cache, bits,
            count, blocks, 5, [], bias, [], [], queryCount, 4, arm64);
        var actual = Search.ScorePlanesQuery(query, centroids, index.Scales, paddedCache, bits,
            count, blocks + extraBlocks, 5, [], bias, [], [], queryCount, 4, arm64);

        Assert.Equal(expected.Indices, actual.Indices);
        Assert.Equal(expected.Scores, actual.Scores);
    }

    [Theory]
    [InlineData(2, "normal", "32705,32581,31761,32014,31935", "41261636,4101bb3b,40e5ad97,40dd2f52,40d55837")]
    [InlineData(2, "zero", "31748,31749,31750,31751,31752", "00000000,00000000,00000000,00000000,00000000")]
    [InlineData(2, "duplicates", "31752,31756,31760,31764,31768", "41bbcf3e,41bbcf3e,41bbcf3e,41bbcf3e,41bbcf3e")]
    [InlineData(4, "normal", "32705,31761,31935,31865,32017", "410bcecd,40dc8dd8,40d98250,40d388ac,40d2a728")]
    [InlineData(4, "zero", "31748,31749,31750,31751,31752", "00000000,00000000,00000000,00000000,00000000")]
    [InlineData(4, "duplicates", "31756,31760,31764,31768,31772", "41bb41c2,41bb41c2,41bb41c2,41bb41c2,41bb41c2")]
    public void StagedSearchSkipsMaskedBlocks(
        int bits, string mode, string expectedIds, string expectedScores)
    {
        const int dim = 64;
        const int count = 32768;
        var vectors = Rows(count, dim, 0xC0FFEEUL);
        var originalQuery = Rows(1, dim, 0xBADF00DUL);
        if (mode == "zero")
        {
            Array.Clear(originalQuery);
        }
        else if (mode == "duplicates")
        {
            var first = vectors.AsSpan(0, dim).ToArray();
            for (var i = 0; i < count; i += 4) first.CopyTo(vectors, i * dim);
            first.CopyTo(originalQuery, 0);
        }

        var index = new TurboQuantIndex(dim, bits);
        index.Add(vectors);
        var (blocked, blocks) = Pack.Repack(index.PackedCodes, count, bits, dim);
        var centroids = Codebook.Compute(bits, dim).Centroids;
        var cache = Search.BuildPlanesCache(blocked, centroids, [], [], bits, dim, count, blocks);
        var mask = new ulong[count / 64];
        for (var i = count - 1024; i < count; i++) mask[i / 64] |= 1UL << (i % 64);
        var query = RotatedQuery(index, originalQuery, out var bias);
        var before = Search.BlocksSkippedByMask;

        var (scores, ids) = Search.ScorePlanesQuery(query, centroids, index.Scales, cache, bits,
            count, blocks, 5, mask, bias, [], [], 1, 1, false);

        AssertResult(ids, scores, expectedIds, expectedScores);
        Assert.True(Search.BlocksSkippedByMask - before >= blocks - 32);
    }

    [Fact]
    public void TwoBitPermuteDotMatchesArmRustBatch()
    {
        const int dim = 64;
        const int count = 129;
        var index = CreateIndex(dim, count, 2, calibrated: true);
        var (blocked, blocks) = Pack.Repack(index.PackedCodes, count, 2, dim);
        var centroids = Codebook.Compute(2, dim).Centroids;
        var mask = new ulong[(count + 63) / 64];
        for (var i = 0; i < count; i += 2) mask[i / 64] |= 1UL << (i % 64);
        var queries = Rows(4, dim, 0xBADF00DUL);
        var allScores = new List<float>();
        var allIds = new List<int>();
        for (var qi = 0; qi < 4; qi++)
        {
            var query = RotatedQuery(index, queries.AsSpan(qi * dim, dim), out var bias);
            var (scores, ids) = Search.Score2BitPermuteDot(query, blocked, centroids, index.Scales,
                dim / 4, count, blocks, 5, mask, bias);
            allScores.AddRange(scores);
            allIds.AddRange(ids);
        }
        AssertResult(allIds, allScores, TwoBitBatchExpectedIds, TwoBitBatchExpectedScores);
    }

    [Fact]
    public void ArmIndexUsesOptInTwoBitPermuteDot()
    {
        const int dim = 64;
        const int count = 129;
        Assert.SkipWhen(!Search.UseArm2BitPermuteDot(dim / 4),
            "This Rust fixture requires the opt-in ARM64 two-bit dot-product path.");
        var index = CreateIndex(dim, count, 2, calibrated: true);
        var queries = Rows(4, dim, 0xBADF00DUL);
        var allowed = Enumerable.Range(0, count).Select(static i => (i & 1) == 0).ToArray();
        var result = index.SearchWithMask(queries, 5, allowed);
        AssertResult(result.Indices.Select(static value => checked((int)value)), result.Scores,
            TwoBitBatchExpectedIds, TwoBitBatchExpectedScores);
    }

    [Theory]
    [InlineData(3, 0)]
    [InlineData(4, 4)]
    [InlineData(7, 4)]
    [InlineData(8, 8)]
    [InlineData(10, 8)]
    [InlineData(12, 12)]
    [InlineData(16, 16)]
    [InlineData(17, 12)]
    [InlineData(20, 20)]
    public void TwoBitArmBatchesKeepRustTailBehavior(int queryCount, int accelerated)
    {
        var actual = Enumerable.Range(0, queryCount).Count(q => Search.IsArmPermuteBatch(q, queryCount));
        Assert.Equal(accelerated, actual);
    }

    [Theory]
    [InlineData(8, 33)]
    [InlineData(64, 129)]
    [InlineData(256, 65)]
    public void ThreeBitSplitAndUnifiedLayoutsHaveIdenticalResults(int dim, int count)
    {
        var index = CreateIndex(dim, count, 3);
        var (blocked, blocks) = Pack.Repack(index.PackedCodes, count, 3, dim);
        var (subCodes, plane, splitBlocks) = Pack.Repack3Bit(index.PackedCodes, count, dim);
        Assert.Equal(blocks, splitBlocks);
        var expanded = new byte[blocked.Length];
        Pack.Expand3BitBlocked(subCodes, plane, dim, blocks, expanded);
        Assert.Equal(blocked, expanded);

        var queries = Rows(3, dim, 0xBADF00DUL);
        var rotation = Rotation.MakeRotationMatrix(dim);
        var centroids = Codebook.Compute(3, dim).Centroids;
        var mask = new ulong[(count + 63) / 64];
        for (var i = 0; i < count; i += 2) mask[i / 64] |= 1UL << (i % 64);
        var unified = Search.RunSearch(queries, 3, rotation, index.PackedCodes, centroids,
            index.Scales, 3, dim, count, 7, mask);
        var split = Search.Search3Bit_(queries, 3, rotation, subCodes.AsSpan(), plane.AsSpan(),
            centroids, index.Scales, dim, count, blocks, 7, mask);
        Assert.Equal(unified.Indices, split.Indices);
        Assert.Equal(unified.Scores, split.Scores);
    }

    [Theory]
    [InlineData(2, 8, 33)]
    [InlineData(2, 64, 129)]
    [InlineData(4, 8, 33)]
    [InlineData(4, 64, 129)]
    public void PlaneCodesReconstructTheBlockedRepresentation(int bits, int dim, int count)
    {
        var index = CreateIndex(dim, count, bits);
        var (blocked, blocks) = Pack.Repack(index.PackedCodes, count, bits, dim);
        var centroids = Codebook.Compute(bits, dim).Centroids;
        var cache = Search.BuildPlanesCache(blocked, centroids, [], [], bits, dim, count, blocks);
        var groups = dim / (8 / bits);
        for (var v = 0; v < count; v++)
        {
            for (var g = 0; g < groups; g++)
            {
                var offset = (v / Constants.Block * groups + g) * Constants.Block + v % Constants.Block;
                Assert.Equal(blocked[offset], Search.ReadPlaneCode(cache, v, bits, g, dim / 8));
            }
        }
    }

    private static TurboQuantIndex CreateIndex(int dim, int count, int bits, bool calibrated = false)
    {
        var index = new TurboQuantIndex(dim, bits);
        if (calibrated) index.Calibrate(Rows(128, dim, 0x51A7EUL));
        index.Add(Rows(count, dim, 0xC0FFEEUL));
        return index;
    }

    private static (TurboQuantIndex Index, float[] Queries) CreateSampleSkewIndex(
        int bits, bool calibrated = false, int queryCount = 2, int shortCount = 1)
    {
        const int dim = 64;
        const int count = 32768;
        var queries = Rows(queryCount, dim, 0xBADF00DUL);
        var vectors = new float[count * dim];
        var stride = count / Constants.Block / 48;
        for (var v = 0; v < count; v++)
        {
            var sample = v / Constants.Block - stride / 2;
            var sampled = sample >= 0 && sample % stride == 0 && sample / stride < 48;
            for (var d = 0; d < dim; d++)
            {
                vectors[v * dim + d] = sampled ? queries[d] : -queries[d];
            }
        }
        for (var qi = 1; qi < queryCount; qi++)
        {
            for (var d = 0; d < dim; d++)
            {
                queries[qi * dim + d] = qi < shortCount ? queries[d] : -queries[d];
            }
        }

        var index = new TurboQuantIndex(dim, bits);
        if (calibrated) index.Calibrate(Rows(128, dim, 0x51A7EUL));
        index.Add(vectors);
        return (index, queries);
    }

    private static float[] Rows(int count, int dim, ulong state)
    {
        var rows = new float[count * dim];
        for (var i = 0; i < rows.Length; i++)
        {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            rows[i] = ((int)((state >> 32) & 0xffff) - 32768) / 32768f;
        }
        return rows;
    }

    private static float[] RotatedQuery(TurboQuantIndex index, ReadOnlySpan<float> input, out float correction)
    {
        var query = input.ToArray();
        new Rotation(index.Dim).Apply(query, new float[index.Dim]);
        var bias = 0d;
        if (!index.TqPlusShift.IsEmpty)
        {
            for (var d = 0; d < index.Dim; d++)
            {
                bias -= (double)query[d] * index.TqPlusShift[d];
                query[d] /= index.TqPlusScale[d];
            }
        }
        correction = (float)bias;
        return query;
    }

    private static string HashScales(ReadOnlySpan<float> scales)
    {
        var bytes = new byte[scales.Length * sizeof(float)];
        for (var i = 0; i < scales.Length; i++)
        {
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(i * sizeof(float)),
                BitConverter.SingleToInt32Bits(scales[i]));
        }
        return Hash(bytes);
    }

    private static string Hash(ReadOnlySpan<byte> bytes)
    {
        var hash = 0xcbf29ce484222325UL;
        foreach (var value in bytes) hash = unchecked((hash ^ value) * 0x100000001b3UL);
        return hash.ToString("x16", CultureInfo.InvariantCulture);
    }

    private static void AssertResult(
        IEnumerable<int> indices, IEnumerable<float> scores, string expectedIds, string expectedScores)
    {
        Assert.Equal(expectedIds.Split(',').Select(static value => int.Parse(value, CultureInfo.InvariantCulture)), indices);
        Assert.Equal(expectedScores.Split(',').Select(static value => uint.Parse(value, NumberStyles.HexNumber, CultureInfo.InvariantCulture)),
            scores.Select(static score => BitConverter.SingleToUInt32Bits(score)));
    }
}
