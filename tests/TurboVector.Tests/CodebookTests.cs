using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TurboVector;

namespace TurboVector.Tests;

public class CodebookTests
{
    [Fact]
    public void CentroidsStrictlyAscending()
    {
        foreach (int bits in new[] { 2, 3, 4 })
        {
            foreach (int dim in new[] { 256, 768, 1536 })
            {
                var (_, centroids) = Codebook.Compute(bits, dim);
                for (int i = 0; i < centroids.Length - 1; i++)
                {
                    Assert.True(
                        centroids[i] < centroids[i + 1],
                        $"centroids not ascending at bits={bits}, dim={dim}: c[{i}]={centroids[i]} >= c[{i + 1}]={centroids[i + 1]}");
                }
            }
        }
    }

    [Fact]
    public void BoundariesStrictlyBetweenCentroids()
    {
        foreach (int bits in new[] { 2, 3, 4 })
        {
            foreach (int dim in new[] { 256, 1536 })
            {
                var (boundaries, centroids) = Codebook.Compute(bits, dim);
                Assert.Equal(centroids.Length - 1, boundaries.Length);
                for (int i = 0; i < boundaries.Length; i++)
                {
                    Assert.True(
                        boundaries[i] > centroids[i],
                        $"boundary[{i}] = {boundaries[i]} not > centroid[{i}] = {centroids[i]} (bits={bits}, dim={dim})");
                    Assert.True(
                        boundaries[i] < centroids[i + 1],
                        $"boundary[{i}] = {boundaries[i]} not < centroid[{i + 1}] = {centroids[i + 1]} (bits={bits}, dim={dim})");
                }
            }
        }
    }

    [Fact]
    public void LevelCountsCorrect()
    {
        foreach (int bits in new[] { 2, 3, 4 })
        {
            var (boundaries, centroids) = Codebook.Compute(bits, 1536);
            Assert.Equal(1 << bits, centroids.Length);
            Assert.Equal((1 << bits) - 1, boundaries.Length);
        }
    }

    [Fact]
    public void SymmetricAboutZero()
    {
        foreach (int bits in new[] { 2, 3, 4 })
        {
            foreach (int dim in new[] { 768, 1536 })
            {
                var (_, centroids) = Codebook.Compute(bits, dim);
                int n = centroids.Length;
                for (int i = 0; i < n / 2; i++)
                {
                    float lo = centroids[i];
                    float hi = centroids[n - 1 - i];
                    Assert.True(
                        MathF.Abs(lo + hi) < 1e-4f,
                        $"asymmetric: c[{i}]={lo} c[{n - 1 - i}]={hi} (bits={bits}, dim={dim})");
                }
            }
        }
    }

    [Fact]
    public void DeterministicForSameParams()
    {
        var (b1, c1) = Codebook.Compute(4, 1536);
        var (b2, c2) = Codebook.Compute(4, 1536);
        Assert.Equal(b1, b2);
        Assert.Equal(c1, c2);
    }

    [Fact]
    public void CentroidsWithinUnitInterval()
    {
        foreach (int bits in new[] { 2, 3, 4 })
        {
            var (_, centroids) = Codebook.Compute(bits, 1536);
            for (int i = 0; i < centroids.Length; i++)
            {
                float c = centroids[i];
                Assert.True(c > -1.0f && c < 1.0f, $"centroid[{i}] = {c} outside (-1, 1) (bits={bits})");
            }
        }
    }
}

internal static class TestHelpers
{
    internal static readonly int[] TailSizes = [32, 33, 63, 64, 65, 96, 127, 128, 129, 160, 191, 192, 193, 256, 257, 500];

    internal static float[] MakeVectors(int n, int dim, ulong seed)
    {
        ulong state = unchecked(seed * 0x9E3779B97F4A7C15UL);
        float[] data = new float[n * dim];
        for (int i = 0; i < data.Length; i++)
        {
            state = unchecked(state * 6364136223846793005UL + 1442695040888963407UL);
            uint bits = ((uint)(state >> 32) & 0x007FFFFFu) | 0x3F800000u;
            float uniform = BitConverter.Int32BitsToSingle((int)bits) - 1.0f;
            data[i] = (uniform * 2.0f) - 1.0f;
        }

        return data;
    }

    internal static float[] RandVec(int dim, ulong seed)
    {
        ulong state = unchecked(seed * 0x9E3779B97F4A7C15UL);
        float[] data = new float[dim];
        for (int i = 0; i < dim; i++)
        {
            state = unchecked(state * 6364136223846793005UL + 1442695040888963407UL);
            uint bits = ((uint)(state >> 32) & 0x007FFFFFu) | 0x3F800000u;
            float uniform = BitConverter.Int32BitsToSingle((int)bits) - 1.0f;
            data[i] = (uniform * 2.0f) - 1.0f;
        }

        return data;
    }

    internal static float[] GaussianNormalized(int n, int dim, ulong seed)
    {
        ulong state = seed | 1UL;

        ulong Next()
        {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            return state;
        }

        float Uniform()
        {
            uint raw = (uint)(Next() >> 40) | 1u;
            return (float)raw / (1u << 24);
        }

        float[] data = new float[n * dim];
        float twoPi = 2.0f * MathF.PI;
        int i = 0;
        while (i < data.Length)
        {
            float u1 = MathF.Max(Uniform(), 1e-7f);
            float u2 = Uniform();
            float r = MathF.Sqrt(-2.0f * MathF.Log(u1));
            float theta = twoPi * u2;
            data[i++] = r * MathF.Cos(theta);
            if (i < data.Length)
            {
                data[i++] = r * MathF.Sin(theta);
            }
        }

        for (int row = 0; row < n; row++)
        {
            int start = row * dim;
            int end = start + dim;
            float norm = 0f;
            for (int j = start; j < end; j++)
            {
                norm += data[j] * data[j];
            }

            norm = MathF.Sqrt(norm);
            if (norm > 0f)
            {
                float inv = 1.0f / norm;
                for (int j = start; j < end; j++)
                {
                    data[j] *= inv;
                }
            }
        }

        return data;
    }

    internal static float[] UnitSphereVectors(int n, int dim, ulong seed)
    {
        ulong state = unchecked(seed * 0x9E3779B97F4A7C15UL);

        float NextU()
        {
            state = unchecked(state * 6364136223846793005UL + 1442695040888963407UL);
            uint bits = ((uint)(state >> 32) & 0x007FFFFFu) | 0x3F800000u;
            return BitConverter.Int32BitsToSingle((int)bits) - 1.0f;
        }

        float[] data = new float[n * dim];
        int idx = 0;
        while (idx < data.Length)
        {
            float u1 = MathF.Max(NextU(), 1e-30f);
            float u2 = NextU();
            float r = MathF.Sqrt(-2.0f * MathF.Log(u1));
            float theta = 2.0f * MathF.PI * u2;
            data[idx++] = r * MathF.Cos(theta);
            if (idx < data.Length)
            {
                data[idx++] = r * MathF.Sin(theta);
            }
        }

        for (int row = 0; row < n; row++)
        {
            int start = row * dim;
            int end = start + dim;
            float norm = 0f;
            for (int j = start; j < end; j++)
            {
                norm += data[j] * data[j];
            }

            norm = MathF.Sqrt(norm);
            if (norm > 1e-10f)
            {
                float inv = 1f / norm;
                for (int j = start; j < end; j++)
                {
                    data[j] *= inv;
                }
            }
        }

        return data;
    }

    internal static float[] MatMul(float[] a, float[] b, int dim)
    {
        float[] output = new float[dim * dim];
        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                float acc = 0f;
                for (int k = 0; k < dim; k++)
                {
                    acc += a[(i * dim) + k] * b[(k * dim) + j];
                }

                output[(i * dim) + j] = acc;
            }
        }

        return output;
    }

    internal static float[] Transpose(float[] matrix, int dim)
    {
        float[] transpose = new float[dim * dim];
        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                transpose[(j * dim) + i] = matrix[(i * dim) + j];
            }
        }

        return transpose;
    }

    internal static float[] MatVec(float[] matrix, float[] vector, int dim)
    {
        float[] output = new float[dim];
        for (int i = 0; i < dim; i++)
        {
            float acc = 0f;
            for (int j = 0; j < dim; j++)
            {
                acc += matrix[(i * dim) + j] * vector[j];
            }

            output[i] = acc;
        }

        return output;
    }

    internal static float L2Norm(float[] vector)
        => MathF.Sqrt(vector.Sum(x => x * x));

    internal static TurboQuantIndex BuildIndex(int n, int dim, ulong seed, int bitWidth = 4)
    {
        float[] data = GaussianNormalized(n, dim, seed);
        var index = new TurboQuantIndex(dim, bitWidth);
        index.Add(data);
        return index;
    }

    internal static (float[] Scores, long[] Indices) ReferenceTopk(TurboQuantIndex idx, float[] query, bool[] mask, int k)
    {
        int n = mask.Length;
        var res = idx.Search(query, n);
        var filtered = new List<(float score, long idx)>();
        for (int i = 0; i < res.K; i++)
        {
            long slot = res.Indices[i];
            if (mask[slot])
            {
                filtered.Add((res.Scores[i], slot));
            }
        }

        if (filtered.Count > k)
        {
            filtered.RemoveRange(k, filtered.Count - k);
        }

        return (filtered.Select(p => p.score).ToArray(), filtered.Select(p => p.idx).ToArray());
    }

    internal static string TempPath(string extension, string prefix)
    {
        string name = Path.ChangeExtension($"{prefix}-{Path.GetRandomFileName()}", extension);
        return Path.Combine(Path.GetTempPath(), name);
    }

    internal static void DeleteFileIfExists(string path)
    {
        if (File.Exists(path))
        {
            File.Delete(path);
        }
    }
}
