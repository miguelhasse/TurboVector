using System.Numerics;

namespace TurboVector;

public static partial class Search
{
    internal sealed class PlanesCache(
        byte[] signs,
        byte[] low,
        float magnitude,
        float alpha,
        float[] beta,
        float alphaFirst,
        float betaFirst,
        float[]? centre)
    {
        public byte[] Signs { get; } = signs;
        public byte[] Low { get; } = low;
        public float Magnitude { get; } = magnitude;
        public float Alpha { get; } = alpha;
        public float[] Beta { get; } = beta;
        public float AlphaFirst { get; } = alphaFirst;
        public float BetaFirst { get; } = betaFirst;
        public float[]? Centre { get; } = centre;
    }

    internal static PlanesCache BuildPlanesCache(
        ReadOnlySpan<byte> blocked,
        ReadOnlySpan<float> centroids,
        ReadOnlySpan<float> shift,
        ReadOnlySpan<float> calibrationScale,
        int bits,
        int dim,
        int nVectors,
        int nBlocks)
    {
        var groups = dim / (8 / bits);
        var signGroups = dim / 8;
        var lowPlanes = bits - 1;
        var signs = new byte[checked(nBlocks * signGroups * Constants.Block)];
        var low = new byte[checked(nVectors * lowPlanes * signGroups)];
        var histogram = new ulong[1 << bits];
        var sampled = Math.Min(nVectors, 64 * Constants.Block);
        var codesPerByte = 8 / bits;

        for (var v = 0; v < nVectors; v++)
        {
            var block = v / Constants.Block;
            var lane = v % Constants.Block;
            for (var group = 0; group < signGroups; group++)
            {
                var signByte = 0;
                var low0 = 0;
                var low1 = 0;
                var low2 = 0;
                for (var j = 0; j < 8; j++)
                {
                    var d = group * 8 + j;
                    var codeByte = blocked[(block * groups + d / codesPerByte) * Constants.Block + lane];
                    var code = (codeByte >> ((codesPerByte - 1 - d % codesPerByte) * bits)) & ((1 << bits) - 1);
                    var bit = 7 - j;
                    signByte |= ((code >> lowPlanes) & 1) << bit;
                    low0 |= (code & 1) << bit;
                    if (lowPlanes > 1) low1 |= ((code >> 1) & 1) << bit;
                    if (lowPlanes > 2) low2 |= ((code >> 2) & 1) << bit;
                    if (v < sampled) histogram[code]++;
                }

                signs[(block * signGroups + group) * Constants.Block + lane] = (byte)signByte;
                var row = v * lowPlanes * signGroups + group;
                low[row] = (byte)low0;
                if (lowPlanes > 1) low[row + signGroups] = (byte)low1;
                if (lowPlanes > 2) low[row + 2 * signGroups] = (byte)low2;
            }
        }

        var beta = new float[3];
        var magnitude = 0f;
        var alpha = 0f;
        var alphaFirst = 0f;
        var betaFirst = 0f;
        if (bits == 2)
        {
            var outer = 0UL;
            for (var v = 0; v < sampled; v++)
            {
                var block = v / Constants.Block;
                var lane = v % Constants.Block;
                for (var group = 0; group < signGroups; group++)
                {
                    var s = signs[(block * signGroups + group) * Constants.Block + lane];
                    var l = low[v * signGroups + group];
                    outer += (uint)BitOperations.PopCount((uint)(byte)~(s ^ l));
                }
            }

            var fraction = (float)outer / (float)(sampled * dim);
            magnitude = centroids[2] * (1f - fraction) + centroids[3] * fraction;
            alpha = (centroids[3] + centroids[2]) * 0.5f;
            beta[0] = (centroids[3] - centroids[2]) * 0.5f;
            alphaFirst = alpha;
            betaFirst = beta[0];
        }
        else
        {
            (magnitude, alpha, alphaFirst, betaFirst) = FitPlaneLevels(histogram, centroids, beta);
        }

        var centre = default(float[]);
        if (!shift.IsEmpty)
        {
            var weights = new float[dim];
            for (var d = 0; d < dim; d++)
            {
                weights[d] = magnitude * shift[d] / calibrationScale[d];
            }

            centre = new float[nVectors];
            for (var v = 0; v < nVectors; v++)
            {
                var block = v / Constants.Block;
                var lane = v % Constants.Block;
                var value = 0f;
                for (var group = 0; group < signGroups; group++)
                {
                    var s = signs[(block * signGroups + group) * Constants.Block + lane];
                    for (var j = 0; j < 8; j++)
                    {
                        var weight = weights[group * 8 + j];
                        value += ((s >> (7 - j)) & 1) != 0 ? weight : -weight;
                    }
                }

                centre[v] = value;
            }
        }

        return new PlanesCache(signs, low, magnitude, alpha, beta, alphaFirst, betaFirst, centre);
    }

    private static (float Magnitude, float Alpha, float AlphaFirst, float BetaFirst) FitPlaneLevels(
        ReadOnlySpan<ulong> histogram,
        ReadOnlySpan<float> centroids,
        Span<float> beta)
    {
        Span<double> matrix = stackalloc double[16];
        Span<double> rhs = stackalloc double[4];
        Span<double> features = stackalloc double[4];
        matrix.Clear();
        rhs.Clear();
        var total = 0d;
        foreach (var count in histogram) total += count;
        total = Math.Max(total, 1d);
        var magnitude = 0d;
        var allWeight = 0d;
        var signTop = 0d;
        var signTarget = 0d;
        var topTarget = 0d;
        for (var code = 0; code < histogram.Length; code++)
        {
            var count = (double)histogram[code];
            var weight = (count + 0.5d) / total;
            var centroid = (double)centroids[code];
            magnitude += (count / total) * Math.Abs(centroid);
            features[0] = (code & 8) != 0 ? 1d : -1d;
            for (var plane = 0; plane < 3; plane++)
            {
                features[plane + 1] = ((code >> plane) & 1) != 0 ? 1d : -1d;
            }

            var sign = features[0];
            var top = features[3];
            allWeight += weight;
            signTop += weight * sign * top;
            signTarget += weight * sign * centroid;
            topTarget += weight * top * centroid;
            for (var a = 0; a < 4; a++)
            {
                rhs[a] += weight * features[a] * centroid;
                for (var b = 0; b < 4; b++)
                {
                    matrix[a * 4 + b] += weight * features[a] * features[b];
                }
            }
        }

        for (var i = 0; i < 4; i++)
        {
            var pivot = i;
            for (var r = i + 1; r < 4; r++)
            {
                if (Math.Abs(matrix[r * 4 + i]) >= Math.Abs(matrix[pivot * 4 + i])) pivot = r;
            }

            if (pivot != i)
            {
                for (var c = 0; c < 4; c++)
                {
                    (matrix[i * 4 + c], matrix[pivot * 4 + c]) = (matrix[pivot * 4 + c], matrix[i * 4 + c]);
                }
                (rhs[i], rhs[pivot]) = (rhs[pivot], rhs[i]);
            }

            var diagonal = matrix[i * 4 + i];
            if (Math.Abs(diagonal) < 1e-12) continue;
            for (var r = 0; r < 4; r++)
            {
                if (r == i) continue;
                var factor = matrix[r * 4 + i] / diagonal;
                for (var c = i; c < 4; c++)
                {
                    matrix[r * 4 + c] -= factor * matrix[i * 4 + c];
                }
                rhs[r] -= factor * rhs[i];
            }
        }

        var alpha = Math.Abs(matrix[0]) < 1e-12 ? 0f : (float)(rhs[0] / matrix[0]);
        for (var j = 0; j < 3; j++)
        {
            var diagonal = matrix[(j + 1) * 4 + j + 1];
            beta[j] = Math.Abs(diagonal) < 1e-12 ? 0f : (float)(rhs[j + 1] / diagonal);
        }

        var determinant = allWeight * allWeight - signTop * signTop;
        var alphaFirst = Math.Abs(determinant) < 1e-12
            ? alpha : (float)((signTarget * allWeight - topTarget * signTop) / determinant);
        var betaFirst = Math.Abs(determinant) < 1e-12
            ? beta[2] : (float)((topTarget * allWeight - signTarget * signTop) / determinant);
        return ((float)magnitude, alpha, alphaFirst, betaFirst);
    }
}
