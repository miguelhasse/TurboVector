namespace TurboVector.Tests;

public class TqPlusCalibrationTests
{
    [Fact]
    public void LazyIndexCanBeCalibratedBeforeItsFirstAdd()
    {
        const int dim = 64;
        float[] sample = TestHelpers.MakeVectors(128, dim, seed: 11);
        var index = TurboQuantIndex.NewLazy(3);

        index.Calibrate2D(sample, dim);
        index.Add2D(sample.AsSpan(0, 3 * dim), dim);

        Assert.Equal(dim, index.Dim);
        Assert.Equal(CalibrationState.Calibrated, index.CalibrationState);
        Assert.Equal(3, index.Len);
        Assert.Equal(dim, index.TqPlusShift.Length);
        Assert.Equal(dim, index.TqPlusScale.Length);
    }

    [Fact]
    public void RecalibrationReencodesExistingRowsAndSearchRemainsUsable()
    {
        const int dim = 64;
        float[] vectors = TestHelpers.MakeVectors(80, dim, seed: 12);
        float[] sample = TestHelpers.MakeVectors(128, dim, seed: 13);
        var index = new TurboQuantIndex(dim, 2);
        index.Add(vectors);
        byte[] before = index.PackedCodes.ToArray();

        index.Calibrate(sample);
        index.Add(vectors.AsSpan(0, dim));

        Assert.Equal(81, index.Len);
        Assert.NotEqual(before, index.PackedCodes.ToArray());
        var results = index.Search(vectors.AsSpan(0, 2 * dim), 5);
        Assert.Equal(2, results.Nq);
        Assert.Equal(5, results.K);
        Assert.All(results.Scores, score => Assert.True(float.IsFinite(score)));
        Assert.All(results.Indices, id => Assert.InRange(id, 0, index.Len - 1));
    }

    [Fact]
    public void CalibratedIndexPersistsCalibrationAndPackedData()
    {
        const int dim = 64;
        float[] sample = TestHelpers.MakeVectors(128, dim, seed: 14);
        var index = new TurboQuantIndex(dim, 4);
        index.Calibrate(sample);

        string path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.tv");
        try
        {
            index.Write(path);
            TurboQuantIndex loaded = TurboQuantIndex.Load(path);

            Assert.Equal(index.CalibrationState, loaded.CalibrationState);
            Assert.Equal(index.TqPlusShift.ToArray(), loaded.TqPlusShift.ToArray());
            Assert.Equal(index.TqPlusScale.ToArray(), loaded.TqPlusScale.ToArray());
            Assert.Equal(index.PackedCodes.ToArray(), loaded.PackedCodes.ToArray());
            Assert.Equal(index.Scales.ToArray(), loaded.Scales.ToArray());
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
        }
    }

    [Fact]
    public void IdMapWrapperUsesTheInnerCalibrationForAddAndSearch()
    {
        const int dim = 64;
        float[] sample = TestHelpers.MakeVectors(128, dim, seed: 15);
        float[] vectors = TestHelpers.MakeVectors(8, dim, seed: 16);
        var index = IdMapIndex.NewLazy(3);

        index.Calibrate2D(sample, dim);
        index.AddWithIds2D(vectors, dim, [41, 42, 43, 44, 45, 46, 47, 48]);
        var (scores, ids) = index.Search(vectors.AsSpan(0, dim), 3);

        Assert.Equal(CalibrationState.Calibrated, index.CalibrationState);
        Assert.Equal(3, ids.Length);
        Assert.All(scores, score => Assert.True(float.IsFinite(score)));
        Assert.All(ids, id => Assert.InRange(id, 41UL, 48UL));
    }

    [Fact]
    public void CalibratedIdMapRoundTripsThroughRustFormat()
    {
        const int dim = 64;
        float[] sample = TestHelpers.MakeVectors(128, dim, seed: 17);
        float[] vectors = TestHelpers.MakeVectors(37, dim, seed: 18);
        ulong[] ids = Enumerable.Range(100, 37).Select(static id => (ulong)id).ToArray();
        var index = IdMapIndex.NewLazy(4);
        index.Calibrate2D(sample, dim);
        index.AddWithIds2D(vectors, dim, ids);

        IdMapIndex loaded = IdMapIndex.Load(index.ToBytes());

        Assert.Equal(index.TqPlusShift.ToArray(), loaded.TqPlusShift.ToArray());
        Assert.Equal(index.TqPlusScale.ToArray(), loaded.TqPlusScale.ToArray());
        var expected = index.Search(vectors.AsSpan(0, dim), 8);
        var actual = loaded.Search(vectors.AsSpan(0, dim), 8);
        Assert.Equal(expected.Ids, actual.Ids);
        Assert.Equal(expected.Scores, actual.Scores);
    }
}
