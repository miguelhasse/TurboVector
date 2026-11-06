namespace TurboVector.Tests;

public class SearchOrderingParityTests
{
    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void EqualScoresAreOrderedByAscendingIndex(int bitWidth)
    {
        const int dim = 8;
        const int vectorCount = 12;
        const int k = 8;
        float[] vector = [0.25f, -0.5f, 0.75f, -1.0f, 1.25f, -1.5f, 1.75f, -2.0f];
        float[] vectors = new float[vectorCount * dim];
        for (int i = 0; i < vectorCount; i++)
        {
            vector.CopyTo(vectors, i * dim);
        }

        var index = new TurboQuantIndex(dim, bitWidth);
        index.Add(vectors);

        SearchResults results = index.Search(vector, k);

        Assert.Equal(Enumerable.Range(0, k).Select(static i => (long)i), results.Indices);

        bool[] mask = Enumerable.Range(0, vectorCount).Select(static i => (i & 1) != 0).ToArray();
        SearchResults masked = index.SearchWithMask([.. vector, .. vector], 3, mask);
        Assert.Equal([1L, 3L, 5L, 1L, 3L, 5L], masked.Indices);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void HigherScoreEvictsHighestIndexAmongTiedMinimums(int bitWidth)
    {
        const int dim = 8;
        float[] low = [-1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f];
        float[] query = [1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f];
        float[] vectors = [.. low, .. low, .. query];
        var index = new TurboQuantIndex(dim, bitWidth);
        index.Add(vectors);

        SearchResults results = index.Search(query, 2);

        Assert.Equal([2L, 0L], results.Indices);
        Assert.True(results.Scores[0] > results.Scores[1]);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void RepeatedReplacementsKeepLowestIndexFromTiedCutoff(int bitWidth)
    {
        const int dim = 8;
        float[] low = [-1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f];
        float[] medium = [0.5f, 0.8660254f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f];
        float[] high = [1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f];
        float[] vectors = [.. low, .. low, .. low, .. high, .. medium];
        var index = new TurboQuantIndex(dim, bitWidth);
        index.Add(vectors);

        SearchResults results = index.Search(high, 3);

        Assert.Equal([3L, 4L, 0L], results.Indices);
        Assert.True(results.Scores[0] > results.Scores[1]);
        Assert.True(results.Scores[1] > results.Scores[2]);
    }

    [Fact]
    public void LowLevelSearchRetainsLegacyOverloadAndAppliesCalibration()
    {
        const int dim = 8;
        const int bitWidth = 2;
        float[] query = [1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f];
        float[] rotation = Rotation.MakeRotationMatrix(dim);
        float[] centroids = Codebook.Compute(bitWidth, dim).Centroids;
        byte[] blockedCodes = new byte[dim / 4 * 32];
        float[] vectorScales = [1.0f];

        (float[] scores, long[] indices) = Search.Search_(
            query, 1, rotation, blockedCodes, centroids, vectorScales,
            bitWidth, dim, 1, 1, 1, null, 0);
        (float[] calibratedScores, long[] calibratedIndices) = Search.Search_(
            query, 1, rotation, blockedCodes, centroids, vectorScales,
            bitWidth, dim, 1, 1, 1, null, 0,
            Enumerable.Repeat(0.1f, dim).ToArray(),
            Enumerable.Repeat(1.0f, dim).ToArray());

        Assert.Equal(indices, calibratedIndices);
        Assert.NotEqual(scores[0], calibratedScores[0]);
    }
}
