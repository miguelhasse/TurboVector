using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using TurboVector;

namespace TurboVector.Tests;

public class KernelCorrectnessTests
{
    [Fact]
    public void SelfQueryReturnsSelfTop1_4Bit()
    {
        const int dim = 512;
        const int bits = 4;

        foreach (int n in TestHelpers.TailSizes)
        {
            float[] data = TestHelpers.GaussianNormalized(n, dim, 0x5EED_0000UL ^ (ulong)n);
            var idx = new TurboQuantIndex(dim, bits);
            idx.Add(data);
            Assert.Equal(n, idx.Len);

            int nq = Math.Min(n, 8);
            float[] queries = data.Take(nq * dim).ToArray();
            SearchResults res = idx.Search(queries, 1);

            for (int qi = 0; qi < nq; qi++)
            {
                long top = res.IndicesForQuery(qi)[0];
                Assert.Equal(qi, (int)top);
            }
        }
    }

    [Fact]
    public void SelfQueryReturnsSelfTop3_2Bit()
    {
        const int dim = 512;
        const int bits = 2;

        foreach (int n in TestHelpers.TailSizes)
        {
            float[] data = TestHelpers.GaussianNormalized(n, dim, 0xC0FF_EE00UL ^ (ulong)n);
            var idx = new TurboQuantIndex(dim, bits);
            idx.Add(data);

            int nq = Math.Min(n, 8);
            float[] queries = data.Take(nq * dim).ToArray();
            int k = Math.Min(3, n);
            SearchResults res = idx.Search(queries, k);

            for (int qi = 0; qi < nq; qi++)
            {
                long[] top = res.IndicesForQuery(qi).ToArray();
                Assert.Contains(qi, top.Select(x => (int)x));
            }
        }
    }

    [Fact]
    public void SearchScoresAreSortedDescending()
    {
        const int dim = 256;
        foreach (int bits in new[] { 2, 3, 4 })
        {
            foreach (int n in new[] { 64, 100, 128, 200, 256, 500 })
            {
                float[] data = TestHelpers.GaussianNormalized(n, dim, 0xA11CEUL ^ (ulong)n ^ (ulong)bits);
                var idx = new TurboQuantIndex(dim, bits);
                idx.Add(data);

                float[] queries = data.Take(4 * dim).ToArray();
                int k = Math.Min(10, n);
                SearchResults res = idx.Search(queries, k);

                for (int qi = 0; qi < 4; qi++)
                {
                    ReadOnlySpan<float> scores = res.ScoresForQuery(qi);
                    for (int i = 0; i < scores.Length - 1; i++)
                    {
                        Assert.True(
                            scores[i] >= scores[i + 1] || !float.IsFinite(scores[i + 1]),
                            $"scores not sorted desc: bits={bits} n={n} qi={qi} [{scores[i]}, {scores[i + 1]}]");
                    }
                }
            }
        }
    }

    [Fact]
    public void SearchIsDeterministicForSameQuery()
    {
        const int dim = 256;
        const int bits = 4;
        foreach (int n in new[] { 64, 65, 127, 128, 129, 500 })
        {
            float[] data = TestHelpers.GaussianNormalized(n, dim, 0xD0D0_D0D0UL ^ (ulong)n);
            var idx = new TurboQuantIndex(dim, bits);
            idx.Add(data);

            float[] queries = data.Take(3 * dim).ToArray();
            SearchResults r1 = idx.Search(queries, Math.Min(10, n));
            SearchResults r2 = idx.Search(queries, Math.Min(10, n));
            Assert.Equal(r1.Indices, r2.Indices);
            Assert.Equal(r1.Scores, r2.Scores);
        }
    }

    [Fact]
    public void SingleQueryMatchesBatchedQuery()
    {
        const int dim = 256;
        const int bits = 4;
        const int n = 500;
        float[] data = TestHelpers.GaussianNormalized(n, dim, 0x1234_5678UL);
        var idx = new TurboQuantIndex(dim, bits);
        idx.Add(data);

        float[] batch = data.Take(5 * dim).ToArray();
        const int k = 10;
        SearchResults batched = idx.Search(batch, k);

        for (int qi = 0; qi < 5; qi++)
        {
            float[] singleQuery = batch.Skip(qi * dim).Take(dim).ToArray();
            SearchResults single = idx.Search(singleQuery, k);
            Assert.Equal(batched.IndicesForQuery(qi).ToArray(), single.IndicesForQuery(0).ToArray());

            ReadOnlySpan<float> batchedScores = batched.ScoresForQuery(qi);
            ReadOnlySpan<float> singleScores = single.ScoresForQuery(0);
            for (int i = 0; i < batchedScores.Length; i++)
            {
                float b = batchedScores[i];
                float s = singleScores[i];
                float tol = Math.Max(1e-5f, 1e-5f * Math.Abs(b));
                Assert.True(Math.Abs(b - s) <= tol, $"single-query vs batched score diff > {tol} at qi={qi} rank={i}: batched={b} single={s}");
            }
        }
    }

    [Fact]
    public async Task ConcurrentSearchMatchesSerial()
    {
        const int dim = 256;
        const int bits = 4;
        const int n = 500;
        float[] data = TestHelpers.GaussianNormalized(n, dim, 0xFACE_CAFEUL);
        var idx = new TurboQuantIndex(dim, bits);
        idx.Add(data);

        float[] queries = TestHelpers.GaussianNormalized(4, dim, 0xBEEF_0000UL);
        SearchResults expected = idx.Search(queries, 10);
        List<long[]> expectedIndices = Enumerable.Range(0, expected.Nq)
            .Select(qi => expected.IndicesForQuery(qi).ToArray())
            .ToList();

        Task[] tasks = Enumerable.Range(0, 8).Select(_ => Task.Run(() =>
        {
            for (int i = 0; i < 16; i++)
            {
                SearchResults result = idx.Search(queries, 10);
                for (int qi = 0; qi < expected.Nq; qi++)
                {
                    Assert.Equal(expectedIndices[qi], result.IndicesForQuery(qi).ToArray());
                }
            }
        })).ToArray();

        await Task.WhenAll(tasks);
    }
}
