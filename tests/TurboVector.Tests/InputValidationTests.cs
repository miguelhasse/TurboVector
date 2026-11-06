namespace TurboVector.Tests;

public class InputValidationTests
{
    [Fact]
    public void AddRejectsNonFiniteAndOversizedCoordinatesWithoutMutatingIndex()
    {
        foreach (float invalid in new[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity, 1e16f, -1e16f })
        {
            var index = new TurboQuantIndex(64, 4);
            float[] vectors = new float[64];
            vectors[3] = invalid;

            ArgumentException exception = Assert.Throws<ArgumentException>(() => index.Add(vectors));

            Assert.Contains("row 0, coordinate 3", exception.Message);
            Assert.Equal(0, index.Len);
        }
    }

    [Fact]
    public void SearchRejectsNonFiniteAndOversizedQueryCoordinates()
    {
        var index = new TurboQuantIndex(64, 4);
        index.Add(TestHelpers.UnitSphereVectors(1, 64, 0x1A11DUL));

        foreach (float invalid in new[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity, 1e16f, -1e16f })
        {
            float[] query = new float[64];
            query[7] = invalid;

            ArgumentException exception = Assert.Throws<ArgumentException>(() => index.Search(query, 1));

            Assert.Contains("row 0, coordinate 7", exception.Message);
        }
    }

    [Fact]
    public void DimensionsAndBitWidthsAreBounded()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new TurboQuantIndex(16_392, 4));
        Assert.Throws<ArgumentOutOfRangeException>(() => Codebook.Compute(5, 64));
        Assert.Throws<ArgumentOutOfRangeException>(() => Codebook.Compute(4, 16_392));
    }
}
