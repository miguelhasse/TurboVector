using System;
using MathNet.Numerics.Distributions;
using TurboVector;

namespace TurboVector.Tests;

public class DistortionTests
{
    private static readonly (int Bits, double PaperMse)[] PaperMse =
    [
        (2, 0.1175),
        (3, 0.03454),
        (4, 0.009497),
    ];

    [Fact]
    public void CodebookMseMatchesPaperAtHighDim()
    {
        const int dim = 1536;
        foreach (var (bits, paperVal) in PaperMse)
        {
            var (boundaries, centroids) = Codebook.Compute(bits, dim);
            double mse = ComputeCodebookMse(boundaries, centroids, dim);
            double expected = paperVal / dim;
            double relErr = Math.Abs(mse - expected) / expected;
            Assert.True(
                relErr < 0.05,
                $"bits={bits}, dim={dim}: codebook MSE={mse:E3} vs Theorem1/d={expected:E3} (rel_err={relErr:F3})");
        }
    }

    [Fact]
    public void CodebookMseWithinShannonFactor()
    {
        foreach (int bits in new[] { 2, 3, 4 })
        {
            foreach (int dim in new[] { 256, 768, 1536 })
            {
                var (boundaries, centroids) = Codebook.Compute(bits, dim);
                double mse = ComputeCodebookMse(boundaries, centroids, dim);
                double shannonBound = Math.Pow(2.0, -2 * bits) / dim;
                double ratio = mse / shannonBound;
                Assert.True(ratio < 3.0, $"bits={bits}, dim={dim}: MSE/Shannon = {ratio:F3} exceeds 3x paper bound");
                Assert.True(ratio > 1.0, $"bits={bits}, dim={dim}: MSE/Shannon = {ratio:F3} below Shannon lower bound");
            }
        }
    }

    [Fact]
    public void PipelineSelfScoreIsUnbiased()
    {
        const int dim = 1536;
        const int n = 500;
        float[] vectors = TestHelpers.UnitSphereVectors(n, dim, 42);

        foreach (var (bits, _) in PaperMse)
        {
            ScoreStats stats = SelfScoreStats(vectors, dim, bits);
            double deficit = Math.Abs(1.0 - stats.Mean);
            Assert.True(
                deficit < 0.005,
                $"bits={bits}: corrected self-score mean = {stats.Mean:F5}, deficit from 1.0 = {deficit:F5}");
        }
    }

    [Fact]
    public void CrossQueryVarianceTightensWithMoreBits()
    {
        const int dim = 512;
        const int n = 200;
        float[] database = TestHelpers.UnitSphereVectors(n, dim, 0);
        float[] queries = TestHelpers.UnitSphereVectors(n, dim, 1);

        ScoreStats s2 = CrossScoreStats(database, queries, dim, 2);
        ScoreStats s4 = CrossScoreStats(database, queries, dim, 4);

        Assert.True(s4.StdDev < s2.StdDev, $"4-bit cross-score stddev {s4.StdDev:F4} not tighter than 2-bit {s2.StdDev:F4}");
    }

    [Fact]
    public void SelfQueryRecallAt1()
    {
        const int dim = 512;
        const int n = 200;
        float[] vectors = TestHelpers.UnitSphereVectors(n, dim, 0);

        var index = new TurboQuantIndex(dim, 4);
        index.Add(vectors);
        index.Prepare();

        int hits = 0;
        for (int i = 0; i < n; i++)
        {
            float[] query = vectors.AsSpan(i * dim, dim).ToArray();
            SearchResults results = index.Search(query, 1);
            if (results.IndicesForQuery(0)[0] == i)
            {
                hits++;
            }
        }

        double recall = (double)hits / n;
        Assert.True(recall >= 0.99, $"recall@1 = {recall:F3} below 0.99 threshold");
    }

    private static double ComputeCodebookMse(float[] boundaries, float[] centroids, int dim)
    {
        double a = (dim - 1.0) / 2.0;
        var beta = new Beta(a, a);
        int n = centroids.Length;
        var edges = new double[n + 1];
        edges[0] = -1.0;
        for (int i = 0; i < boundaries.Length; i++)
        {
            edges[i + 1] = boundaries[i];
        }

        edges[n] = 1.0;

        double mse = 0;
        for (int i = 0; i < n; i++)
        {
            double lo = edges[i];
            double hi = edges[i + 1];
            double c = centroids[i];
            mse += Simpson(x => (x - c) * (x - c) * beta.Density((x + 1.0) / 2.0) / 2.0, lo, hi, 4000);
        }

        return mse;
    }

    private static double Simpson(Func<double, double> f, double a, double b, int n)
    {
        n &= ~1;
        double h = (b - a) / n;
        double sum = f(a) + f(b);
        for (int i = 1; i < n; i++)
        {
            double x = a + (i * h);
            sum += (i % 2 == 0) ? 2.0 * f(x) : 4.0 * f(x);
        }

        return sum * h / 3.0;
    }

    private static ScoreStats SelfScoreStats(float[] vectors, int dim, int bits)
    {
        int n = vectors.Length / dim;
        var index = new TurboQuantIndex(dim, bits);
        index.Add(vectors);
        index.Prepare();

        double[] scores = new double[n];
        for (int i = 0; i < n; i++)
        {
            float[] query = vectors.AsSpan(i * dim, dim).ToArray();
            SearchResults results = index.Search(query, 1);
            scores[i] = results.ScoresForQuery(0)[0];
        }

        double mean = 0;
        for (int i = 0; i < scores.Length; i++)
        {
            mean += scores[i];
        }

        mean /= n;

        double variance = 0;
        for (int i = 0; i < scores.Length; i++)
        {
            variance += Math.Pow(scores[i] - mean, 2);
        }

        variance /= n;
        return new ScoreStats(mean, Math.Sqrt(variance));
    }

    private static ScoreStats CrossScoreStats(float[] database, float[] queries, int dim, int bits)
    {
        int nQueries = queries.Length / dim;
        var index = new TurboQuantIndex(dim, bits);
        index.Add(database);
        index.Prepare();

        double[] scores = new double[nQueries];
        for (int i = 0; i < nQueries; i++)
        {
            float[] query = queries.AsSpan(i * dim, dim).ToArray();
            SearchResults results = index.Search(query, 1);
            scores[i] = results.ScoresForQuery(0)[0];
        }

        double mean = 0;
        for (int i = 0; i < scores.Length; i++)
        {
            mean += scores[i];
        }

        mean /= nQueries;

        double variance = 0;
        for (int i = 0; i < scores.Length; i++)
        {
            variance += Math.Pow(scores[i] - mean, 2);
        }

        variance /= nQueries;
        return new ScoreStats(mean, Math.Sqrt(variance));
    }

    private readonly record struct ScoreStats(double Mean, double StdDev);
}
