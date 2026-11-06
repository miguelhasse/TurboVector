using System;
using System.Linq;
using System.Threading.Tasks;
using TurboVector;

namespace TurboVector.Tests;

public class ConcurrentSearchTests
{
    private static TurboQuantIndex BuildIndex()
    {
        const int dim = 256;
        const int bitWidth = 4;
        const int n = 1024;

        float[] vectors = TestHelpers.MakeVectors(n, dim, 1);
        var index = new TurboQuantIndex(dim, bitWidth);
        index.Add(vectors);
        Assert.Equal(n, index.Len);
        return index;
    }

    [Fact]
    public async Task SearchIsDeterministicAcrossThreads()
    {
        TurboQuantIndex index = BuildIndex();
        index.Prepare();

        float[] queries = TestHelpers.MakeVectors(4, index.Dim, 42);
        const int k = 10;
        SearchResults reference = index.Search(queries, k);
        long[][] refIndices = Enumerable.Range(0, reference.Nq)
            .Select(qi => reference.IndicesForQuery(qi).ToArray())
            .ToArray();

        Task[] tasks = Enumerable.Range(0, 16).Select(_ => Task.Run(() =>
        {
            for (int i = 0; i < 32; i++)
            {
                SearchResults result = index.Search(queries, k);
                Assert.Equal(reference.Nq, result.Nq);
                Assert.Equal(k, result.K);
                for (int qi = 0; qi < refIndices.Length; qi++)
                {
                    Assert.Equal(refIndices[qi], result.IndicesForQuery(qi).ToArray());
                }
            }
        })).ToArray();

        await Task.WhenAll(tasks);
    }

    [Fact]
    public async Task LazyInitIsSafeWhenPrepareIsSkipped()
    {
        TurboQuantIndex index = BuildIndex();
        float[] queries = TestHelpers.MakeVectors(2, index.Dim, 7);
        const int k = 5;

        Task[] tasks = Enumerable.Range(0, 32).Select(_ => Task.Run(() =>
        {
            SearchResults result = index.Search(queries, k);
            Assert.Equal(result.Nq * result.K, result.Indices.Length);
            Assert.Equal(result.Nq * result.K, result.Scores.Length);
        })).ToArray();

        await Task.WhenAll(tasks);
    }

    [Fact]
    public async Task PrepareIsIdempotentFromMultipleThreads()
    {
        TurboQuantIndex index = BuildIndex();

        Task[] tasks = Enumerable.Range(0, 8).Select(_ => Task.Run(() =>
        {
            index.Prepare();
            index.Prepare();
        })).ToArray();

        await Task.WhenAll(tasks);

        float[] queries = TestHelpers.MakeVectors(1, index.Dim, 99);
        SearchResults result = index.Search(queries, 3);
        Assert.Equal(3, result.K);
    }

    [Fact]
    public void AddAfterSearchInvalidatesBlockedCache()
    {
        TurboQuantIndex index = BuildIndex();
        float[] queries = TestHelpers.MakeVectors(1, index.Dim, 3);
        _ = index.Search(queries, 5);

        float[] more = TestHelpers.MakeVectors(512, index.Dim, 11);
        index.Add(more);
        Assert.Equal(1536, index.Len);

        SearchResults after = index.Search(queries, 5);
        foreach (long idx in after.IndicesForQuery(0))
        {
            Assert.True(idx >= 0, "negative index");
            Assert.True(idx < index.Len, "stale index out of range");
        }
    }

    [Fact]
    public void WriteLoadPreservesConcurrentSearchResults()
    {
        TurboQuantIndex index = BuildIndex();
        float[] queries = TestHelpers.MakeVectors(3, index.Dim, 123);
        const int k = 8;

        SearchResults before = index.Search(queries, k);
        string path = TestHelpers.TempPath(".tv", "concurrent-roundtrip");
        try
        {
            index.Write(path);
            TurboQuantIndex reloaded = TurboQuantIndex.Load(path);

            Assert.Equal(index.Len, reloaded.Len);
            Assert.Equal(index.Dim, reloaded.Dim);
            Assert.Equal(index.BitWidth, reloaded.BitWidth);

            SearchResults after = reloaded.Search(queries, k);
            for (int qi = 0; qi < before.Nq; qi++)
            {
                Assert.Equal(before.IndicesForQuery(qi).ToArray(), after.IndicesForQuery(qi).ToArray());
            }
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
        }
    }
}
