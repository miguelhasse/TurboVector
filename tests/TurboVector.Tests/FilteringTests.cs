using System;
using System.Collections.Generic;
using System.Linq;
using TurboVector;

namespace TurboVector.Tests;

public class FilteringTests
{
    [Fact]
    public void MaskMatchesPostHocFilter()
    {
        const int dim = 128;
        const int n = 256;
        TurboQuantIndex idx = TestHelpers.BuildIndex(n, dim, 0xF11D_0001UL);
        float[] query = TestHelpers.GaussianNormalized(1, dim, 0xF11D_0002UL);

        bool[] mask = new bool[n];
        for (int i = 0; i < n; i += 2)
        {
            mask[i] = true;
        }

        SearchResults masked = idx.SearchWithMask(query, 10, mask);
        var (refScores, refIndices) = TestHelpers.ReferenceTopk(idx, query, mask, 10);

        Assert.Equal(10, masked.K);
        Assert.Equal(refScores, masked.Scores);
        Assert.Equal(refIndices, masked.Indices);
        foreach (long slot in masked.Indices)
        {
            Assert.True(mask[slot], $"kernel returned disallowed slot {slot}");
        }
    }

    [Fact]
    public void MaskNoneEqualsMaskAllTrue()
    {
        const int dim = 64;
        const int n = 200;
        TurboQuantIndex idx = TestHelpers.BuildIndex(n, dim, 0xF11D_0003UL);
        float[] query = TestHelpers.GaussianNormalized(1, dim, 0xF11D_0004UL);

        SearchResults unfiltered = idx.Search(query, 20);
        bool[] allTrue = Enumerable.Repeat(true, n).ToArray();
        SearchResults filtered = idx.SearchWithMask(query, 20, allTrue);

        Assert.Equal(unfiltered.K, filtered.K);
        Assert.Equal(unfiltered.Scores, filtered.Scores);
        Assert.Equal(unfiltered.Indices, filtered.Indices);
    }

    [Fact]
    public void EffectiveKShrinksWhenAllowlistSmallerThanK()
    {
        const int dim = 64;
        const int n = 100;
        TurboQuantIndex idx = TestHelpers.BuildIndex(n, dim, 0xF11D_0005UL);
        float[] query = TestHelpers.GaussianNormalized(1, dim, 0xF11D_0006UL);

        bool[] mask = new bool[n];
        mask[3] = true;
        mask[42] = true;
        mask[77] = true;

        SearchResults res = idx.SearchWithMask(query, 10, mask);
        Assert.Equal(3, res.K);
        Assert.Equal(3, res.Scores.Length);
        Assert.Equal(3, res.Indices.Length);
        foreach (long slot in res.Indices)
        {
            Assert.True(mask[slot]);
        }
    }

    [Fact]
    public void AllFalseMaskReturnsEmptyResults()
    {
        const int dim = 64;
        const int n = 64;
        TurboQuantIndex idx = TestHelpers.BuildIndex(n, dim, 0xF11D_0007UL);
        float[] query = TestHelpers.GaussianNormalized(1, dim, 0xF11D_0008UL);

        bool[] mask = new bool[n];
        SearchResults res = idx.SearchWithMask(query, 5, mask);
        Assert.Equal(0, res.K);
        Assert.Empty(res.Scores);
        Assert.Empty(res.Indices);
    }

    [Fact]
    public void MaskLengthMismatchPanics()
    {
        const int dim = 64;
        const int n = 50;
        TurboQuantIndex idx = TestHelpers.BuildIndex(n, dim, 0xF11D_0009UL);
        float[] query = TestHelpers.GaussianNormalized(1, dim, 0xF11D_000AUL);

        var ex = Assert.Throws<ArgumentException>(() => idx.SearchWithMask(query, 5, new bool[10]));
        Assert.Contains("mask length", ex.Message);
    }

    [Fact]
    public void MultiQueryBatchRespectsMask()
    {
        const int dim = 128;
        const int n = 256;
        const int nq = 7;
        TurboQuantIndex idx = TestHelpers.BuildIndex(n, dim, 0xF11D_000BUL);
        float[] queries = TestHelpers.GaussianNormalized(nq, dim, 0xF11D_000CUL);

        bool[] mask = new bool[n];
        for (int i = 0; i < n; i++)
        {
            if (i % 3 == 0)
            {
                mask[i] = true;
            }
        }

        SearchResults res = idx.SearchWithMask(queries, 8, mask);
        Assert.Equal(nq, res.Nq);
        Assert.Equal(8, res.K);
        Assert.Equal(nq * 8, res.Indices.Length);

        for (int qi = 0; qi < nq; qi++)
        {
            long[] row = res.IndicesForQuery(qi).ToArray();
            float[] scoreRow = res.ScoresForQuery(qi).ToArray();
            foreach (long slot in row)
            {
                Assert.True(mask[slot], $"query {qi}: kernel returned disallowed slot {slot}");
            }

            for (int i = 0; i < scoreRow.Length - 1; i++)
            {
                Assert.True(scoreRow[i] >= scoreRow[i + 1], $"query {qi}: scores not descending: [{scoreRow[i]}, {scoreRow[i + 1]}]");
            }

            float[] queryRow = queries.AsSpan(qi * dim, dim).ToArray();
            var (refScores, refIndices) = TestHelpers.ReferenceTopk(idx, queryRow, mask, res.K);
            Assert.Equal(refIndices, row);
            for (int i = 0; i < scoreRow.Length; i++)
            {
                float a = scoreRow[i];
                float b = refScores[i];
                float tol = 1e-4f * Math.Max(Math.Max(Math.Abs(a), Math.Abs(b)), 1.0f);
                Assert.True(Math.Abs(a - b) <= tol, $"query {qi}: score {a} vs reference {b}");
            }
        }
    }

    [Fact]
    public void AllowlistReturnsOnlyListedIds()
    {
        const int dim = 128;
        const int n = 100;
        float[] data = TestHelpers.GaussianNormalized(n, dim, 0xF11D_1001UL);
        ulong[] ids = Enumerable.Range(0, n).Select(i => 1000UL + (ulong)i).ToArray();
        var idx = new IdMapIndex(dim, 4);
        idx.AddWithIds(data, ids);

        float[] query = TestHelpers.GaussianNormalized(1, dim, 0xF11D_1002UL);
        ulong[] allowed = [1003UL, 1010UL, 1042UL, 1077UL, 1099UL];
        var (scores, returnedIds) = idx.SearchWithAllowlist(query, 10, allowed);

        Assert.Equal(allowed.Length, scores.Length);
        Assert.Equal(allowed.Length, returnedIds.Length);
        foreach (ulong id in returnedIds)
        {
            Assert.Contains(id, allowed);
        }
    }

    [Fact]
    public void AllowlistNoneEquivalentToPlainSearch()
    {
        const int dim = 64;
        const int n = 80;
        float[] data = TestHelpers.GaussianNormalized(n, dim, 0xF11D_1003UL);
        ulong[] ids = Enumerable.Range(0, n).Select(i => 7000UL + ((ulong)i * 13UL)).ToArray();
        var idx = new IdMapIndex(dim, 4);
        idx.AddWithIds(data, ids);

        float[] query = TestHelpers.GaussianNormalized(1, dim, 0xF11D_1004UL);
        var (s1, i1) = idx.Search(query, 5);
        var (s2, i2) = idx.SearchWithAllowlist(query, 5, null);
        Assert.Equal(s1, s2);
        Assert.Equal(i1, i2);
    }

    [Fact]
    public void EmptyAllowlistPanics()
    {
        const int dim = 64;
        float[] data = TestHelpers.GaussianNormalized(10, dim, 0xF11D_1005UL);
        ulong[] ids = Enumerable.Range(0, 10).Select(i => (ulong)i).ToArray();
        var idx = new IdMapIndex(dim, 4);
        idx.AddWithIds(data, ids);

        float[] query = TestHelpers.GaussianNormalized(1, dim, 0xF11D_1006UL);
        var ex = Assert.Throws<ArgumentException>(() => idx.SearchWithAllowlist(query, 3, Array.Empty<ulong>()));
        Assert.Contains("allowlist is empty", ex.Message);
    }

    [Fact]
    public void UnknownIdInAllowlistPanics()
    {
        const int dim = 64;
        float[] data = TestHelpers.GaussianNormalized(10, dim, 0xF11D_1007UL);
        ulong[] ids = Enumerable.Range(0, 10).Select(i => (ulong)i).ToArray();
        var idx = new IdMapIndex(dim, 4);
        idx.AddWithIds(data, ids);

        float[] query = TestHelpers.GaussianNormalized(1, dim, 0xF11D_1008UL);
        var ex = Assert.Throws<ArgumentException>(() => idx.SearchWithAllowlist(query, 3, [5UL, 999UL]));
        Assert.Contains("not present in index", ex.Message);
    }

    [Fact]
    public void BlockSkipAtOnePercentSelectivityMatchesPostFilter()
    {
        const int dim = 128;
        const int n = 4096;
        float[] data = TestHelpers.GaussianNormalized(n, dim, 0xB10C_5417UL);
        var idx = new TurboQuantIndex(dim, 4);
        idx.Add(data);
        idx.Prepare();

        int nAllowed = n / 100;
        int[] allowedSlots = Enumerable.Range(0, nAllowed).Select(i => (i * 97) % n).ToArray();
        bool[] mask = new bool[n];
        foreach (int slot in allowedSlots)
        {
            mask[slot] = true;
        }

        float[] query = TestHelpers.GaussianNormalized(1, dim, 0xB10C_5418UL);
        const int k = 8;
        SearchResults masked = idx.SearchWithMask(query, k, mask);
        SearchResults dense = idx.Search(query, n);

        List<(float Score, long Index)> expected = new();
        ReadOnlySpan<long> denseIds = dense.IndicesForQuery(0);
        ReadOnlySpan<float> denseScores = dense.ScoresForQuery(0);
        for (int i = 0; i < denseIds.Length; i++)
        {
            if (mask[denseIds[i]])
            {
                expected.Add((denseScores[i], denseIds[i]));
            }
        }

        expected.Sort((a, b) => b.Score.CompareTo(a.Score));
        if (expected.Count > k)
        {
            expected.RemoveRange(k, expected.Count - k);
        }

        long[] maskedIds = masked.IndicesForQuery(0).ToArray();
        float[] maskedScores = masked.ScoresForQuery(0).ToArray();
        Assert.Equal(expected.Count, maskedIds.Length);
        for (int i = 0; i < expected.Count; i++)
        {
            Assert.Equal(expected[i].Index, maskedIds[i]);
            Assert.True(Math.Abs(maskedScores[i] - expected[i].Score) < 1e-4f, $"rank {i}: score mismatch (got {maskedScores[i]}, want {expected[i].Score})");
        }
    }

    [Fact]
    public void BlockSkipAtExtremeSelectivityReturnsOnlyAllowed()
    {
        const int dim = 64;
        const int n = 8192;
        float[] data = TestHelpers.GaussianNormalized(n, dim, 0xB10C_5419UL);
        var idx = new TurboQuantIndex(dim, 4);
        idx.Add(data);
        idx.Prepare();

        int[] allowedSlots = [17, 533, 1024, 2500, 6700, 8000];
        bool[] mask = new bool[n];
        foreach (int slot in allowedSlots)
        {
            mask[slot] = true;
        }

        float[] query = TestHelpers.GaussianNormalized(1, dim, 0xB10C_541AUL);
        SearchResults results = idx.SearchWithMask(query, 4, mask);
        long[] ids = results.IndicesForQuery(0).ToArray();

        Assert.Equal(4, ids.Length);
        foreach (long id in ids)
        {
            Assert.Contains((int)id, allowedSlots);
        }
    }

    [Fact]
    public void BlockSkipPathActuallyFiresUnderSelectiveMask()
    {
        const int dim = 64;
        const int n = 4096;
        float[] data = TestHelpers.GaussianNormalized(n, dim, 0xC0DE_5417UL);
        var idx = new TurboQuantIndex(dim, 4);
        idx.Add(data);
        idx.Prepare();

        bool[] mask = new bool[n];
        for (int slot = n - 40; slot < n; slot++)
        {
            mask[slot] = true;
        }

        float[] query = TestHelpers.GaussianNormalized(1, dim, 0xC0DE_5418UL);
        Search.ResetBlocksSkippedByMask();
        long before = Search.BlocksSkippedByMask;
        _ = idx.SearchWithMask(query, 8, mask);
        long after = Search.BlocksSkippedByMask;
        long delta = after - before;

        Assert.True(delta > 0, $"block-skip counter did not increment during selective search (before={before}, after={after})");
        Assert.True(delta >= 50, $"block-skip fired only {delta} times for a search where most blocks should be empty");
    }

    [Fact]
    public void BlockSkipWithAllSlotsAllowedMatchesUnmasked()
    {
        const int dim = 64;
        const int n = 1024;
        float[] data = TestHelpers.GaussianNormalized(n, dim, 0xB10C_541BUL);
        var idx = new TurboQuantIndex(dim, 4);
        idx.Add(data);
        idx.Prepare();

        bool[] mask = Enumerable.Repeat(true, n).ToArray();
        float[] query = TestHelpers.GaussianNormalized(1, dim, 0xB10C_541CUL);
        const int k = 16;

        SearchResults withMask = idx.SearchWithMask(query, k, mask);
        SearchResults noMask = idx.Search(query, k);

        Assert.Equal(noMask.IndicesForQuery(0).ToArray(), withMask.IndicesForQuery(0).ToArray());
        Assert.Equal(noMask.ScoresForQuery(0).ToArray(), withMask.ScoresForQuery(0).ToArray());
    }

    [Fact]
    public void AllowlistSurvivesSwapRemove()
    {
        const int dim = 64;
        const int n = 30;
        float[] data = TestHelpers.GaussianNormalized(n, dim, 0xF11D_1009UL);
        ulong[] ids = Enumerable.Range(0, n).Select(i => 5000UL + (ulong)i).ToArray();
        var idx = new IdMapIndex(dim, 4);
        idx.AddWithIds(data, ids);

        ulong[] allowed = [5005UL, 5015UL, 5020UL];
        float[] query = TestHelpers.GaussianNormalized(1, dim, 0xF11D_100AUL);

        _ = idx.SearchWithAllowlist(query, 3, allowed);
        Assert.True(idx.Remove(5025UL));
        var after = idx.SearchWithAllowlist(query, 3, allowed);
        Assert.Equal(3, after.Ids.Length);
        foreach (ulong id in after.Ids)
        {
            Assert.Contains(id, allowed);
        }
    }
}
