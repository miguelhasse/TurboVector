using System;
using TurboVector;

namespace TurboVector.Tests;

public class RotationTests
{
    [Fact]
    public void OrthogonalAcrossDims()
    {
        foreach (int dim in new[] { 32, 64, 128, 256 })
        {
            float[] rotation = Rotation.MakeRotationMatrix(dim);
            float[] transpose = TestHelpers.Transpose(rotation, dim);
            float[] product = TestHelpers.MatMul(transpose, rotation, dim);
            for (int i = 0; i < dim; i++)
            {
                for (int j = 0; j < dim; j++)
                {
                    float expected = i == j ? 1f : 0f;
                    float got = product[(i * dim) + j];
                    Assert.True(
                        MathF.Abs(got - expected) < 1e-3f,
                        $"R^T R[{i}][{j}] = {got}, expected {expected} (dim={dim})");
                }
            }
        }
    }

    [Fact]
    public void PreservesNorm()
    {
        foreach (int dim in new[] { 32, 64, 128 })
        {
            float[] rotation = Rotation.MakeRotationMatrix(dim);
            for (ulong seed = 0; seed < 5; seed++)
            {
                float[] x = TestHelpers.RandVec(dim, seed);
                float[] y = TestHelpers.MatVec(rotation, x, dim);
                float nx = TestHelpers.L2Norm(x);
                float ny = TestHelpers.L2Norm(y);
                Assert.True(
                    MathF.Abs(nx - ny) / nx < 1e-3f,
                    $"norm changed: |Rx|={ny} vs |x|={nx} (dim={dim}, seed={seed})");
            }
        }
    }

    [Fact]
    public void DeterministicForSameDim()
    {
        float[] r1 = Rotation.MakeRotationMatrix(128);
        float[] r2 = Rotation.MakeRotationMatrix(128);
        Assert.Equal(r1.Length, r2.Length);
        for (int i = 0; i < r1.Length; i++)
        {
            Assert.Equal(r1[i], r2[i]);
        }
    }

    [Fact]
    public void InverseRoundTripViaTranspose()
    {
        const int dim = 128;
        float[] rotation = Rotation.MakeRotationMatrix(dim);
        float[] transpose = TestHelpers.Transpose(rotation, dim);
        for (ulong seed = 0; seed < 5; seed++)
        {
            float[] x = TestHelpers.RandVec(dim, seed);
            float[] rx = TestHelpers.MatVec(rotation, x, dim);
            float[] xHat = TestHelpers.MatVec(transpose, rx, dim);
            for (int j = 0; j < dim; j++)
            {
                Assert.True(
                    MathF.Abs(x[j] - xHat[j]) < 1e-3f,
                    $"R^T R x [{j}] = {xHat[j]} vs original {x[j]} (seed={seed})");
            }
        }
    }

    [Fact]
    public void SizeMatchesDimSquared()
    {
        foreach (int dim in new[] { 16, 64, 256 })
        {
            float[] rotation = Rotation.MakeRotationMatrix(dim);
            Assert.Equal(dim * dim, rotation.Length);
        }
    }
}
