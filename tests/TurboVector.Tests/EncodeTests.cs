using System;
using TurboVector;

namespace TurboVector.Tests;

public class EncodeTests
{
    [Fact]
    public void ProducesExpectedShape()
    {
        foreach (int bitWidth in new[] { 2, 4 })
        {
            const int dim = 128;
            const int n = 17;
            float[] rotation = Rotation.MakeRotationMatrix(dim);
            var (boundaries, centroids) = Codebook.Compute(bitWidth, dim);
            float[] vectors = TestHelpers.MakeVectors(n, dim, 0);

            var (packed, scales) = Encoder.Encode(vectors, n, dim, rotation, boundaries, centroids, bitWidth);

            int bytesPerRow = bitWidth * (dim / 8);
            Assert.Equal(n * bytesPerRow, packed.Length);
            Assert.Equal(n, scales.Length);
        }
    }

    [Fact]
    public void ScalesSatisfyRabitqIdentity()
    {
        const int dim = 128;
        const int n = 10;
        float[] rotation = Rotation.MakeRotationMatrix(dim);
        var (boundaries, centroids) = Codebook.Compute(4, dim);
        float[] vectors = TestHelpers.MakeVectors(n, dim, 0);

        var (_, scales) = Encoder.Encode(vectors, n, dim, rotation, boundaries, centroids, 4);

        for (int i = 0; i < n; i++)
        {
            ReadOnlySpan<float> row = vectors.AsSpan(i * dim, dim);
            float norm = 0f;
            for (int j = 0; j < dim; j++)
            {
                norm += row[j] * row[j];
            }

            norm = MathF.Sqrt(norm);
            float invNorm = norm > 0f ? 1.0f / norm : 0f;
            float[] rotated = new float[dim];
            for (int k = 0; k < dim; k++)
            {
                float acc = 0f;
                for (int j = 0; j < dim; j++)
                {
                    acc += rotation[(k * dim) + j] * row[j] * invNorm;
                }

                rotated[k] = acc;
            }

            double inner = 0.0;
            for (int k = 0; k < dim; k++)
            {
                int code = 0;
                for (int b = 0; b < boundaries.Length; b++)
                {
                    if (rotated[k] > boundaries[b])
                    {
                        code++;
                    }
                }

                inner += rotated[k] * centroids[code];
            }

            double expectedScale = norm / Math.Max(inner, 1e-10);
            double relErr = Math.Abs(scales[i] - expectedScale) / Math.Max(Math.Abs(expectedScale), 1e-10);
            Assert.True(relErr < 1e-4, $"scale identity broken at i={i}: stored={scales[i]}, expected={expectedScale}, rel_err={relErr}");
        }
    }

    [Fact]
    public void DeterministicOutput()
    {
        const int dim = 128;
        const int n = 5;
        float[] rotation = Rotation.MakeRotationMatrix(dim);
        var (boundaries, centroids) = Codebook.Compute(4, dim);
        float[] vectors = TestHelpers.MakeVectors(n, dim, 0);

        var (p1, s1) = Encoder.Encode(vectors, n, dim, rotation, boundaries, centroids, 4);
        var (p2, s2) = Encoder.Encode(vectors, n, dim, rotation, boundaries, centroids, 4);

        Assert.Equal(p1, p2);
        Assert.Equal(s1, s2);
    }

    [Fact]
    public void HandlesZeroVector()
    {
        const int dim = 128;
        float[] rotation = Rotation.MakeRotationMatrix(dim);
        var (boundaries, centroids) = Codebook.Compute(4, dim);
        float[] zeros = new float[dim];

        var (packed, scales) = Encoder.Encode(zeros, 1, dim, rotation, boundaries, centroids, 4);

        Assert.Equal(0.0f, scales[0]);
        Assert.True(float.IsFinite(scales[0]));
        Assert.Equal(4 * (dim / 8), packed.Length);
    }
}
