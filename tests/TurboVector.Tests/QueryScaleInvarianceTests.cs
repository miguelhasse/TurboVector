namespace TurboVector.Tests;

public class QueryScaleInvarianceTests
{
    [Theory]
    [InlineData(2, 1, false)]
    [InlineData(3, 1, false)]
    [InlineData(4, 1, false)]
    [InlineData(2, 3, false)]
    [InlineData(3, 3, false)]
    [InlineData(4, 3, false)]
    [InlineData(2, 1, true)]
    [InlineData(3, 1, true)]
    [InlineData(4, 1, true)]
    [InlineData(2, 3, true)]
    [InlineData(3, 3, true)]
    [InlineData(4, 3, true)]
    public void PowerOfTwoQueryScalingPreservesRanksAndScalesScores(int bitWidth, int queryCount, bool masked)
    {
        const int dim = 64;
        const int top = 10;
        var index = new TurboQuantIndex(dim, bitWidth);
        index.Add(TestHelpers.UnitSphereVectors(128, dim, 0xC0FFEEUL));
        var queries = TestHelpers.UnitSphereVectors(queryCount, dim, 0xBADF00DUL);
        var mask = masked
            ? Enumerable.Range(0, 128).Select(static value => (value & 1) == 0).ToArray()
            : Array.Empty<bool>();
        var baseline = index.SearchWithMask(queries, top, mask);

        foreach (var exponent in new[] { -20, -40, -60, -100 })
        {
            var factor = MathF.ScaleB(1f, exponent);
            var scaledQueries = queries.Select(value => value * factor).ToArray();
            var scaled = index.SearchWithMask(scaledQueries, top, mask);

            Assert.Equal(baseline.Indices, scaled.Indices);
            for (var i = 0; i < baseline.Scores.Length; i++)
            {
                Assert.Equal(baseline.Scores[i] * factor, scaled.Scores[i]);
            }
        }
    }
}
