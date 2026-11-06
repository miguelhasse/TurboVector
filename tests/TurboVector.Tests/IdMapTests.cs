using System;
using System.IO;
using System.Linq;
using TurboVector;

namespace TurboVector.Tests;

public class IdMapTests
{
    [Fact]
    public void AddWithIdsUpdatesLenAndContains()
    {
        const int dim = 128;
        float[] data = TestHelpers.GaussianNormalized(5, dim, 0xA11D_0000UL);
        var idx = new IdMapIndex(dim, 4);
        idx.AddWithIds(data, [100UL, 200UL, 300UL, 400UL, 500UL]);

        Assert.Equal(5, idx.Len);
        Assert.True(idx.Contains(300UL));
        Assert.False(idx.Contains(999UL));
    }

    [Fact]
    public void SearchReturnsIdsNotSlots()
    {
        const int dim = 256;
        float[] data = TestHelpers.GaussianNormalized(10, dim, 0xA11D_0001UL);
        var idx = new IdMapIndex(dim, 4);
        ulong[] ids = Enumerable.Range(0, 10).Select(i => 1_000_000UL + (ulong)i).ToArray();
        idx.AddWithIds(data, ids);

        for (int i = 0; i < ids.Length; i++)
        {
            float[] query = data.AsSpan(i * dim, dim).ToArray();
            var (_, gotIds) = idx.Search(query, 1);
            Assert.Equal(ids[i], gotIds[0]);
        }
    }

    [Fact]
    public void RemoveReturnsFalseForMissingId()
    {
        const int dim = 128;
        float[] data = TestHelpers.GaussianNormalized(3, dim, 0xA11D_0002UL);
        var idx = new IdMapIndex(dim, 4);
        idx.AddWithIds(data, [1UL, 2UL, 3UL]);

        Assert.False(idx.Remove(999UL));
        Assert.Equal(3, idx.Len);
    }

    [Fact]
    public void RemoveExistingIdShrinksAndHidesIt()
    {
        const int dim = 256;
        float[] data = TestHelpers.GaussianNormalized(10, dim, 0xA11D_0003UL);
        var idx = new IdMapIndex(dim, 4);
        ulong[] ids = Enumerable.Range(0, 10).Select(i => ((ulong)i * 7UL) + 11UL).ToArray();
        idx.AddWithIds(data, ids);

        ulong targetId = ids[2];
        Assert.True(idx.Remove(targetId));
        Assert.Equal(9, idx.Len);
        Assert.False(idx.Contains(targetId));

        float[] query = data.AsSpan(2 * dim, dim).ToArray();
        var (_, gotIds) = idx.Search(query, 9);
        Assert.DoesNotContain(targetId, gotIds);
    }

    [Fact]
    public void RemainingIdsStillSelfQueryAfterMixedRemoves()
    {
        const int dim = 384;
        float[] data = TestHelpers.GaussianNormalized(20, dim, 0xA11D_0004UL);
        var idx = new IdMapIndex(dim, 4);
        ulong[] ids = Enumerable.Range(0, 20).Select(i => ((ulong)i * 100UL) + 5UL).ToArray();
        idx.AddWithIds(data, ids);

        idx.Remove(ids[7]);
        idx.Remove(ids[19]);
        idx.Remove(ids[0]);

        Assert.Equal(17, idx.Len);
        Assert.False(idx.Contains(ids[7]));
        Assert.False(idx.Contains(ids[19]));
        Assert.False(idx.Contains(ids[0]));

        for (int i = 0; i < ids.Length; i++)
        {
            if (i is 0 or 7 or 19)
            {
                continue;
            }

            float[] query = data.AsSpan(i * dim, dim).ToArray();
            var (_, gotIds) = idx.Search(query, 1);
            Assert.Equal(ids[i], gotIds[0]);
        }
    }

    [Fact]
    public void RemoveThenReAddSameIdIsAllowed()
    {
        const int dim = 128;
        float[] data = TestHelpers.GaussianNormalized(5, dim, 0xA11D_0005UL);
        var idx = new IdMapIndex(dim, 4);
        idx.AddWithIds(data, [1UL, 2UL, 3UL, 4UL, 5UL]);

        Assert.True(idx.Remove(3UL));
        Assert.False(idx.Contains(3UL));

        float[] newVec = TestHelpers.GaussianNormalized(1, dim, 0xA11D_BEEFUL);
        idx.AddWithIds(newVec, [3UL]);
        Assert.True(idx.Contains(3UL));
        Assert.Equal(5, idx.Len);
    }

    [Fact]
    public void AddWithIdsRejectsDuplicateId()
    {
        const int dim = 128;
        float[] data = TestHelpers.GaussianNormalized(5, dim, 0xA11D_0006UL);
        var idx = new IdMapIndex(dim, 4);
        idx.AddWithIds(data.AsSpan(0, 2 * dim), [1UL, 2UL]);

        var ex = Assert.Throws<InvalidOperationException>(() => idx.AddWithIds(data.AsSpan(2 * dim, dim), [2UL]));
        Assert.Contains("already present", ex.Message);
    }

    [Fact]
    public void AddWithIdsRejectsLengthMismatch()
    {
        const int dim = 128;
        float[] data = TestHelpers.GaussianNormalized(5, dim, 0xA11D_0007UL);
        var idx = new IdMapIndex(dim, 4);

        var ex = Assert.Throws<ArgumentException>(() => idx.AddWithIds(data, [1UL, 2UL, 3UL]));
        Assert.True(ex.Message.Contains("expected", StringComparison.OrdinalIgnoreCase), ex.Message);
    }

    [Fact]
    public void WriteAndLoadRoundTrips()
    {
        const int dim = 256;
        float[] data = TestHelpers.GaussianNormalized(10, dim, 0xA11D_0100UL);
        ulong[] ids = Enumerable.Range(0, 10).Select(i => 2000UL + (ulong)i).ToArray();

        var idx = new IdMapIndex(dim, 4);
        idx.AddWithIds(data, ids);
        idx.Remove(2003UL);
        idx.Remove(2007UL);

        string path = TestHelpers.TempPath(".tvim", "turbovec-idmap-roundtrip");
        try
        {
            idx.Write(path);
            IdMapIndex restored = IdMapIndex.Load(path);
            Assert.Equal(8, restored.Len);
            Assert.True(restored.Contains(2000UL));
            Assert.False(restored.Contains(2003UL));
            Assert.False(restored.Contains(2007UL));

            for (int i = 0; i < ids.Length; i++)
            {
                ulong id = ids[i];
                if (id is 2003UL or 2007UL)
                {
                    continue;
                }

                float[] query = data.AsSpan(i * dim, dim).ToArray();
                var (_, gotIds) = restored.Search(query, 1);
                Assert.Equal(id, gotIds[0]);
            }
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
        }
    }

    [Fact]
    public void LoadRejectsWrongMagic()
    {
        string path = TestHelpers.TempPath(".tvim", "turbovec-idmap-badmagic");
        try
        {
            File.WriteAllBytes(path, [.. "XXXX\x01"u8.ToArray()]);
            Assert.ThrowsAny<Exception>(() => IdMapIndex.Load(path));
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
        }
    }

    [Fact]
    public void EmptyIndexRoundTrip()
    {
        const int dim = 128;
        var idx = new IdMapIndex(dim, 4);
        string path = TestHelpers.TempPath(".tvim", "turbovec-idmap-empty");

        try
        {
            idx.Write(path);
            IdMapIndex restored = IdMapIndex.Load(path);
            Assert.Equal(0, restored.Len);
            Assert.Equal(dim, restored.Dim);
            Assert.Equal(4, restored.BitWidth);
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
        }
    }
}
