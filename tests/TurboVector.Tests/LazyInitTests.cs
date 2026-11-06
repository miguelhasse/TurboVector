using System;
using TurboVector;

namespace TurboVector.Tests;

public class LazyInitTests
{
    private const int Dim = 64;

    [Fact]
    public void NewLazyStartsWithNoDim()
    {
        var idx = TurboQuantIndex.NewLazy(4);
        Assert.Null(idx.DimOpt);
        Assert.Equal(0, idx.Dim);
        Assert.Equal(0, idx.Len);
        Assert.Equal(4, idx.BitWidth);
    }

    [Fact]
    public void Add2DLocksDimOnFirstCall()
    {
        var idx = TurboQuantIndex.NewLazy(4);
        float[] data = TestHelpers.UnitSphereVectors(3, Dim, 0xA00D_0001UL);
        idx.Add2D(data, Dim);
        Assert.Equal(Dim, idx.DimOpt);
        Assert.Equal(3, idx.Len);
    }

    [Fact]
    public void Add2DSubsequentCallsMustMatchDim()
    {
        var idx = TurboQuantIndex.NewLazy(4);
        idx.Add2D(TestHelpers.UnitSphereVectors(2, Dim, 0xA00D_0002UL), Dim);
        idx.Add2D(TestHelpers.UnitSphereVectors(2, Dim, 0xA00D_0003UL), Dim);
        Assert.Equal(4, idx.Len);
    }

    [Fact]
    public void Add2DPanicsOnDimChange()
    {
        var idx = TurboQuantIndex.NewLazy(4);
        idx.Add2D(TestHelpers.UnitSphereVectors(1, Dim, 0xA00D_0004UL), Dim);
        float[] wrong = TestHelpers.UnitSphereVectors(1, Dim * 2, 0xA00D_0005UL);
        var ex = Assert.Throws<ArgumentException>(() => idx.Add2D(wrong, Dim * 2));
        Assert.Contains("dim mismatch", ex.Message);
    }

    [Fact]
    public void PlainAddPanicsOnLazyUncommitted()
    {
        var idx = TurboQuantIndex.NewLazy(4);
        float[] data = TestHelpers.UnitSphereVectors(1, Dim, 0xA00D_0006UL);
        var ex = Assert.Throws<InvalidOperationException>(() => idx.Add(data));
        Assert.Contains("dim is not set", ex.Message);
    }

    [Fact]
    public void SearchOnLazyUncommittedReturnsEmpty()
    {
        var idx = TurboQuantIndex.NewLazy(4);
        float[] queries = TestHelpers.UnitSphereVectors(2, Dim, 0xA00D_0007UL);
        SearchResults res = idx.Search(queries, 5);
        Assert.Empty(res.Scores);
        Assert.Empty(res.Indices);
        Assert.Equal(0, res.K);
    }

    [Fact]
    public void PrepareOnLazyUncommittedIsNoop()
    {
        var idx = TurboQuantIndex.NewLazy(4);
        idx.Prepare();
    }

    [Fact]
    public void WriteLoadRoundTripLazyUncommitted()
    {
        string path = TestHelpers.TempPath(".tv", "lazy-uncommitted");
        try
        {
            TurboQuantIndex.NewLazy(4).Write(path);
            TurboQuantIndex loaded = TurboQuantIndex.Load(path);
            Assert.Null(loaded.DimOpt);
            Assert.Equal(0, loaded.Len);
            Assert.Equal(4, loaded.BitWidth);
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
        }
    }

    [Fact]
    public void WriteLoadRoundTripEagerIndexStillWorks()
    {
        string path = TestHelpers.TempPath(".tv", "lazy-eager");
        try
        {
            var idx = new TurboQuantIndex(Dim, 4);
            idx.Add(TestHelpers.UnitSphereVectors(4, Dim, 0xA00D_0008UL));
            idx.Write(path);

            TurboQuantIndex loaded = TurboQuantIndex.Load(path);
            Assert.Equal(Dim, loaded.DimOpt);
            Assert.Equal(4, loaded.Len);
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
        }
    }

    [Fact]
    public void WriteLoadRoundTripLazyAfterCommittedAdd()
    {
        string path = TestHelpers.TempPath(".tv", "lazy-committed");
        try
        {
            var idx = TurboQuantIndex.NewLazy(2);
            idx.Add2D(TestHelpers.UnitSphereVectors(3, Dim, 0xA00D_0009UL), Dim);
            idx.Write(path);

            TurboQuantIndex loaded = TurboQuantIndex.Load(path);
            Assert.Equal(Dim, loaded.DimOpt);
            Assert.Equal(3, loaded.Len);
            Assert.Equal(2, loaded.BitWidth);
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
        }
    }

    [Fact]
    public void IdMapNewLazyStartsWithNoDim()
    {
        var idx = IdMapIndex.NewLazy(4);
        Assert.Null(idx.DimOpt);
        Assert.Equal(0, idx.Dim);
        Assert.Equal(0, idx.Len);
    }

    [Fact]
    public void IdMapAddWithIds2DLocksDim()
    {
        var idx = IdMapIndex.NewLazy(4);
        float[] data = TestHelpers.UnitSphereVectors(3, Dim, 0xA00D_0010UL);
        idx.AddWithIds2D(data, Dim, [10UL, 20UL, 30UL]);
        Assert.Equal(Dim, idx.DimOpt);
        Assert.Equal(3, idx.Len);
        Assert.True(idx.Contains(20UL));
    }

    [Fact]
    public void IdMapPlainAddWithIdsPanicsOnLazyUncommitted()
    {
        var idx = IdMapIndex.NewLazy(4);
        float[] data = TestHelpers.UnitSphereVectors(1, Dim, 0xA00D_0011UL);
        var ex = Assert.Throws<InvalidOperationException>(() => idx.AddWithIds(data, [42UL]));
        Assert.Contains("dim is not set", ex.Message);
    }

    [Fact]
    public void IdMapSearchOnLazyUncommittedReturnsEmpty()
    {
        var idx = IdMapIndex.NewLazy(4);
        float[] queries = TestHelpers.UnitSphereVectors(1, Dim, 0xA00D_0012UL);
        var (scores, ids) = idx.Search(queries, 5);
        Assert.Empty(scores);
        Assert.Empty(ids);
    }

    [Fact]
    public void IdMapWriteLoadRoundTripLazyUncommitted()
    {
        string path = TestHelpers.TempPath(".tvim", "idmap-lazy-uncommitted");
        try
        {
            IdMapIndex.NewLazy(2).Write(path);
            IdMapIndex loaded = IdMapIndex.Load(path);
            Assert.Null(loaded.DimOpt);
            Assert.Equal(0, loaded.Len);
            Assert.Equal(2, loaded.BitWidth);
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
        }
    }

    [Fact]
    public void IdMapWriteLoadRoundTripLazyAfterCommittedAdd()
    {
        string path = TestHelpers.TempPath(".tvim", "idmap-lazy-committed");
        ulong[] ids = [100UL, 200UL, 300UL];
        try
        {
            var idx = IdMapIndex.NewLazy(4);
            idx.AddWithIds2D(TestHelpers.UnitSphereVectors(3, Dim, 0xA00D_0013UL), Dim, ids);
            idx.Write(path);

            IdMapIndex loaded = IdMapIndex.Load(path);
            Assert.Equal(Dim, loaded.DimOpt);
            Assert.Equal(3, loaded.Len);
            foreach (ulong id in ids)
            {
                Assert.True(loaded.Contains(id));
            }
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
        }
    }
}
