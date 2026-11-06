namespace TurboVector;

/// <summary>
/// Lloyd-Max scalar quantizer for the Beta distribution.
/// After orthogonal rotation, each coordinate of a unit vector on S^(d-1)
/// follows Beta((d-1)/2, (d-1)/2) on [-1, 1]. This module computes optimal
/// quantization boundaries and centroids for that distribution.
/// </summary>
public static class Codebook
{
    private const double GammaR = 10.900511;
    private const double F64Precision = 2.22044604925031308085e-16;
    private const double F64MinPositive = 2.2250738585072014e-308;
    private const double LnTwoSqrtEOverPi = 0.6207822376352452223455184457816472;
    private static readonly double[] GammaCoefficients =
    [
        2.48574089138753565546e-5, 1.05142378581721974210, -3.45687097222016235469,
        4.51227709466894823700, -2.98285225323576655721, 1.05639711577126713077,
        -1.95428773191645869583e-1, 1.70970543404441224307e-2, -5.71926117404305781283e-4,
        4.63399473359905636708e-6, -2.71994908488607703910e-9,
    ];

    /// <summary>Returns quantization boundaries and centroids for the given bit width and dimension.</summary>
    public static (float[] Boundaries, float[] Centroids) Compute(int bits, int dim)
    {
        InputValidation.ValidateBitWidth(bits, nameof(bits));
        InputValidation.ValidateDimension(dim, nameof(dim));
        return LloydMax(bits, dim, 200, 1e-12);
    }

    internal static double BetaCdf(double x, double shape)
        => RegularizedBetaIncomplete(x, shape, shape);

    private static (float[] Boundaries, float[] Centroids) LloydMax(int bits, int dim, int maxIter, double tol)
    {
        double a = (dim - 1.0) / 2.0;
        int nLevels = 1 << bits;
        double logBetaAA = LogGamma(a + a) - LogGamma(a) - LogGamma(a);

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
                    double mean = AdaptiveSimpson(
                        x =>
                        {
                            double t = (x + 1.0) / 2.0;
                            return x * BetaDensityHalf(t, a, logBetaAA) / 2.0;
                        },
                        lo,
                        hi,
                        1e-14,
                        50);
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

        float[] resultCentroids = new float[nLevels];
        for (int i = 0; i < nLevels; i++)
        {
            resultCentroids[i] = (float)centroids[i];
        }

        float[] resultBoundaries = new float[nLevels - 1];
        for (int i = 0; i < nLevels - 1; i++)
        {
            resultBoundaries[i] = (resultCentroids[i] + resultCentroids[i + 1]) * 0.5f;
        }

        return (resultBoundaries, resultCentroids);
    }

    private static double LogGamma(double x)
    {
        if (x < 0.5)
        {
            double sum = GammaCoefficients[0];
            for (int i = 1; i < GammaCoefficients.Length; i++)
            {
                sum += GammaCoefficients[i] / (i - x);
            }

            return Math.Log(Math.PI) - Math.Log(Math.Sin(Math.PI * x)) - Math.Log(sum)
                - LnTwoSqrtEOverPi - (0.5 - x) * Math.Log((0.5 - x + GammaR) / Math.E);
        }

        double s = GammaCoefficients[0];
        for (int i = 1; i < GammaCoefficients.Length; i++)
        {
            s += GammaCoefficients[i] / (x + i - 1.0);
        }

        return Math.Log(s) + LnTwoSqrtEOverPi
            + (x - 0.5) * Math.Log((x - 0.5 + GammaR) / Math.E);
    }

    private static double BetaDensityHalf(double x, double a, double logBetaAA)
    {
        if (x <= 0.0 || x >= 1.0)
        {
            return 0.0;
        }

        double bb = (a - 1.0) * Math.Log(x);
        double cc = (a - 1.0) * Math.Log(1.0 - x);
        return Math.Exp(logBetaAA + bb + cc);
    }

    private static double RegularizedBetaIncomplete(double x, double a, double b)
    {
        if (x == 0.0)
        {
            return 0.0;
        }

        if (x == 1.0)
        {
            return 1.0;
        }

        double bt = Math.Exp(
            LogGamma(a + b) - LogGamma(a) - LogGamma(b)
            + a * Math.Log(x) + b * Math.Log(1.0 - x));
        bool symmetricTransform = x >= (a + 1.0) / (a + b + 2.0);
        double fpMin = F64MinPositive / F64Precision;
        if (symmetricTransform)
        {
            (a, b) = (b, a);
            x = 1.0 - x;
        }

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

        for (int m = 1; m < 141; m++)
        {
            double md = m;
            double m2 = md * 2.0;
            double aa = md * (b - md) * x / ((qam + m2) * (a + m2));
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
            h = h * d * c;

            aa = -(a + md) * (qab + md) * x / ((a + m2) * (qap + m2));
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

            if (Math.Abs(del - 1.0) <= F64Precision)
            {
                return symmetricTransform ? 1.0 - bt * h / a : bt * h / a;
            }
        }

        return symmetricTransform ? 1.0 - bt * h / a : bt * h / a;
    }

    private static double AdaptiveSimpson(Func<double, double> f, double a, double b, double tolerance, int maxDepth)
    {
        double mid = (a + b) / 2.0;
        double fa = f(a);
        double fb = f(b);
        double fm = f(mid);
        double whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb);
        return AdaptiveSimpsonRecursive(f, a, b, fa, fb, fm, whole, tolerance, maxDepth);
    }

    private static double AdaptiveSimpsonRecursive(
        Func<double, double> f,
        double a,
        double b,
        double fa,
        double fb,
        double fm,
        double whole,
        double tolerance,
        int depth)
    {
        double mid = (a + b) / 2.0;
        double m1 = (a + mid) / 2.0;
        double m2 = (mid + b) / 2.0;
        double fm1 = f(m1);
        double fm2 = f(m2);
        double left = (mid - a) / 6.0 * (fa + 4.0 * fm1 + fm);
        double right = (b - mid) / 6.0 * (fm + 4.0 * fm2 + fb);
        double refined = left + right;

        if (depth == 0 || Math.Abs(refined - whole) < 15.0 * tolerance)
        {
            return refined + (refined - whole) / 15.0;
        }

        return AdaptiveSimpsonRecursive(f, a, mid, fa, fm, fm1, left, tolerance / 2.0, depth - 1)
            + AdaptiveSimpsonRecursive(f, mid, b, fm, fb, fm2, right, tolerance / 2.0, depth - 1);
    }
}
