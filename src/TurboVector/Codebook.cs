namespace TurboVector;

/// <summary>
/// Lloyd-Max scalar quantizer for the Beta distribution.
/// After orthogonal rotation, each coordinate of a unit vector on S^(d-1)
/// follows Beta((d-1)/2, (d-1)/2) on [-1, 1]. This module computes optimal
/// quantization boundaries and centroids for that distribution.
/// </summary>
public static class Codebook
{
    private static readonly double[] LanczosCoefficients =
    [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
        12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
    ];

    private static readonly double[] GlNodes =
    [
        -0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.4333953941292472, -0.1488743389816312,
         0.1488743389816312,  0.4333953941292472,  0.6794095682990244,  0.8650633666889845,  0.9739065285171717
    ];

    private static readonly double[] GlWeights =
    [
        0.0666713443086881, 0.1494513491505806, 0.2190863625159820, 0.2692667193099963, 0.2955242247147529,
        0.2955242247147529, 0.2692667193099963, 0.2190863625159820, 0.1494513491505806, 0.0666713443086881
    ];

    /// <summary>Returns quantization boundaries and centroids for the given bit width and dimension.</summary>
    public static (float[] Boundaries, float[] Centroids) Compute(int bits, int dim)
        => LloydMax(bits, dim, 200, 1e-12);

    private static (float[] Boundaries, float[] Centroids) LloydMax(int bits, int dim, int maxIter, double tol)
    {
        double a = (dim - 1.0) / 2.0;
        double logBetaAA = LogBeta(a, a);

        int nLevels = 1 << bits;

        double stdDev = Math.Sqrt(2.0 * a / ((2.0 * a + 1.0) * 4.0 * a));
        double spread = 3.0 * stdDev;
        double[] centroids = new double[nLevels];

        for (int i = 0; i < nLevels; i++)
        {
            centroids[i] = -spread + (2.0 * spread * i / (nLevels - 1.0));
        }

        // Pre-allocate arrays once outside the iteration loop (reused via swap).
        double[] boundaries = new double[nLevels - 1];
        double[] edges = new double[nLevels + 1];
        double[] newCentroids = new double[nLevels];
        edges[0] = -1.0;
        edges[nLevels] = 1.0;

        double DensityIntegrand(double x)
        {
            double t = (x + 1.0) / 2.0;
            return x * BetaDensityHalf(t, a, logBetaAA) / 2.0;
        }

        for (int iter = 0; iter < maxIter; iter++)
        {
            for (int i = 0; i < nLevels - 1; i++)
            {
                boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0;
            }

            for (int i = 0; i < boundaries.Length; i++)
            {
                edges[i + 1] = boundaries[i];
            }

            for (int i = 0; i < nLevels; i++)
            {
                double lo = edges[i];
                double hi = edges[i + 1];
                double cdfLo = RegularizedBetaIncomplete((lo + 1.0) / 2.0, a, a);
                double cdfHi = RegularizedBetaIncomplete((hi + 1.0) / 2.0, a, a);
                double prob = cdfHi - cdfLo;

                if (prob < 1e-15)
                {
                    newCentroids[i] = centroids[i];
                }
                else
                {
                    double mean = GaussLegendre10(DensityIntegrand, lo, hi);
                    newCentroids[i] = mean / prob;
                }
            }

            double maxChange = 0.0;
            for (int i = 0; i < nLevels; i++)
            {
                maxChange = Math.Max(maxChange, Math.Abs(centroids[i] - newCentroids[i]));
            }

            // Swap buffers so the newly computed centroids become current without allocating.
            (centroids, newCentroids) = (newCentroids, centroids);

            if (maxChange < tol)
            {
                break;
            }
        }

        float[] resultBoundaries = new float[nLevels - 1];

        for (int i = 0; i < nLevels - 1; i++)
        {
            resultBoundaries[i] = (float)((centroids[i] + centroids[i + 1]) / 2.0);
        }

        float[] resultCentroids = new float[nLevels];

        for (int i = 0; i < nLevels; i++)
        {
            resultCentroids[i] = (float)centroids[i];
        }

        return (resultBoundaries, resultCentroids);
    }

    private static double LogGamma(double x)
    {
        if (x < 0.5)
        {
            return Math.Log(Math.PI / Math.Sin(Math.PI * x)) - LogGamma(1.0 - x);
        }

        x -= 1.0;
        const double g = 7.0;
        double a = LanczosCoefficients[0];

        for (int i = 1; i < LanczosCoefficients.Length; i++)
        {
            a += LanczosCoefficients[i] / (x + i);
        }

        double t = x + g + 0.5;
        return 0.5 * Math.Log(2.0 * Math.PI) + (x + 0.5) * Math.Log(t) - t + Math.Log(a);
    }

    private static double LogBeta(double a, double b)
        => LogGamma(a) + LogGamma(b) - LogGamma(a + b);

    private static double BetaDensityHalf(double x, double a, double logBetaAA)
    {
        if (x <= 0.0 || x >= 1.0)
        {
            return 0.0;
        }

        return Math.Exp((a - 1.0) * Math.Log(x) + (a - 1.0) * Math.Log(1.0 - x) - logBetaAA);
    }

    private static double RegularizedBetaIncomplete(double x, double a, double b)
    {
        if (x <= 0.0)
        {
            return 0.0;
        }

        if (x >= 1.0)
        {
            return 1.0;
        }

        if (x > (a + 1.0) / (a + b + 2.0))
        {
            return 1.0 - RegularizedBetaIncomplete(1.0 - x, b, a);
        }

        double lbetaAb = LogBeta(a, b);
        double front = Math.Exp(Math.Log(x) * a + Math.Log(1.0 - x) * b - lbetaAb) / a;
        return front * BetaContinuedFraction(x, a, b);
    }

    private static double BetaContinuedFraction(double x, double a, double b)
    {
        const int maxIter = 200;
        const double eps = 3e-15;
        const double fpMin = 1e-30;

        double qab = a + b;
        double qap = a + 1.0;
        double qam = a - 1.0;
        double c = 1.0;
        double d = 1.0 - qab * x / qap;
        if (Math.Abs(d) < fpMin)
        {
            d = fpMin;
        }

        d = 1.0 / d;
        double h = d;

        for (int m = 1; m <= maxIter; m++)
        {
            int m2 = 2 * m;
            double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d;

            if (Math.Abs(d) < fpMin)
            {
                d = fpMin;
            }

            c = 1.0 + aa / c;

            if (Math.Abs(c) < fpMin)
            {
                c = fpMin;
            }

            d = 1.0 / d;
            h *= d * c;

            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d;

            if (Math.Abs(d) < fpMin)
            {
                d = fpMin;
            }

            c = 1.0 + aa / c;
            if (Math.Abs(c) < fpMin)
            {
                c = fpMin;
            }

            d = 1.0 / d;
            double del = d * c;
            h *= del;

            if (Math.Abs(del - 1.0) < eps)
            {
                break;
            }
        }

        return h;
    }

    private static double GaussLegendre10(Func<double, double> f, double lo, double hi)
    {
        const int panels = 4;
        double width = (hi - lo) / panels;
        double total = 0.0;

        for (int panel = 0; panel < panels; panel++)
        {
            double panelLo = lo + (panel * width);
            double panelHi = panelLo + width;
            double mid = 0.5 * (panelLo + panelHi);
            double half = 0.5 * (panelHi - panelLo);
            double sum = 0.0;
            for (int i = 0; i < 10; i++)
            {
                sum += GlWeights[i] * f(mid + half * GlNodes[i]);
            }

            total += half * sum;
        }

        return total;
    }
}
