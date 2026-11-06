using System;
using System.Linq;
using TurboVector;

namespace TurboVector.Tests;

public class SwapRemoveTests
{
    [Fact]
    public void SwapRemoveShrinksLengthAndReturnsLastIndex()
    {
        const int dim = 256;
        float[] data = TestHelpers.GaussianNormalized(10, dim, 0xDE1E_7E00UL);
        var idx = new TurboQuantIndex(dim, 4);
        idx.Add(data);
        Assert.Equal(10, idx.Len);

        int movedFrom = idx.SwapRemove(3);
        Assert.Equal(9, movedFrom);
        Assert.Equal(9, idx.Len);
    }

    [Fact]
    public void SwapRemoveLastIsNoSwap()
    {
        const int dim = 256;
        float[] data = TestHelpers.GaussianNormalized(5, dim, 0xDE1E_7E01UL);
        var idx = new TurboQuantIndex(dim, 4);
        idx.Add(data);

        int movedFrom = idx.SwapRemove(4);
        Assert.Equal(4, movedFrom);
        Assert.Equal(4, idx.Len);
    }

    [Fact]
    public void SearchAfterSwapRemoveReflectsNewLayout()
    {
        const int dim = 512;
        const int n = 100;
        float[] data = TestHelpers.GaussianNormalized(n, dim, 0xDE1E_7E02UL);
        var idx = new TurboQuantIndex(dim, 4);
        idx.Add(data);

        float[] q = data.AsSpan(5 * dim, dim).ToArray();
        SearchResults res = idx.Search(q, 1);
        Assert.Equal(5L, res.IndicesForQuery(0)[0]);

        int movedFrom = idx.SwapRemove(5);
        Assert.Equal(n - 1, movedFrom);
        Assert.Equal(n - 1, idx.Len);

        float[] movedQuery = data.AsSpan((n - 1) * dim, dim).ToArray();
        SearchResults movedRes = idx.Search(movedQuery, 1);
        Assert.Equal(5L, movedRes.IndicesForQuery(0)[0]);
    }

    [Fact]
    public void DeletedVectorNoLongerReturned()
    {
        const int dim = 512;
        const int n = 64;
        float[] data = TestHelpers.GaussianNormalized(n, dim, 0xDE1E_7E03UL);
        var idx = new TurboQuantIndex(dim, 4);
        idx.Add(data);

        idx.SwapRemove(7);
        float[] query = data.AsSpan(7 * dim, dim).ToArray();
        SearchResults res = idx.Search(query, idx.Len);
        long[] indices = res.IndicesForQuery(0).ToArray();
        Assert.Equal(idx.Len, indices.Length);
        Assert.True(!indices.Contains(7L) || indices[0] != 7L, "deleted vector appears as top-1 after swap_remove");
    }

    [Fact]
    public void RemainingVectorsStillSelfQueryCorrectly()
    {
        const int dim = 384;
        const int n = 80;
        float[] data = TestHelpers.GaussianNormalized(n, dim, 0xDE1E_7E04UL);
        var idx = new TurboQuantIndex(dim, 4);
        idx.Add(data);

        int[] liveAtSlot = Enumerable.Range(0, n).ToArray();
        foreach (int toDelete in new[] { 10, 5, 40, 0 })
        {
            int last = liveAtSlot.Length - 1;
            _ = idx.SwapRemove(toDelete);
            (liveAtSlot[toDelete], liveAtSlot[last]) = (liveAtSlot[last], liveAtSlot[toDelete]);
            Array.Resize(ref liveAtSlot, last);
        }

        for (int slot = 0; slot < liveAtSlot.Length; slot++)
        {
            int original = liveAtSlot[slot];
            float[] query = data.AsSpan(original * dim, dim).ToArray();
            SearchResults res = idx.Search(query, 1);
            Assert.Equal(slot, (int)res.IndicesForQuery(0)[0]);
        }
    }

    [Fact]
    public void SwapRemoveOutOfBoundsPanics()
    {
        const int dim = 128;
        float[] data = TestHelpers.GaussianNormalized(3, dim, 0xDE1E_7E05UL);
        var idx = new TurboQuantIndex(dim, 4);
        idx.Add(data);

        var ex = Assert.Throws<ArgumentOutOfRangeException>(() => idx.SwapRemove(3));
        Assert.Contains("out of bounds", ex.Message);
    }
}
