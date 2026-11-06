using System.Numerics;

namespace TurboVector;

public enum CalibrationState
{
    Uncalibrated,
    Calibrated,
}

internal static class TqPlusCalibration
{
    internal static (float[] Shift, float[] Scale) Fit(
        ReadOnlySpan<float> sample,
        int n,
        int dim,
        Rotation rotation,
        ReadOnlySpan<float> centroids)
    {
        float[] rotated = new float[checked(n * dim)];
        float[] scratch = new float[dim];
        for (int row = 0; row < n; row++)
        {
            ReadOnlySpan<float> source = sample.Slice(row * dim, dim);
            float norm = Encoder.ComputeNorm(source);
            float inverseNorm = norm > 1e-10f ? 1.0f / norm : 0.0f;
            rotation.ApplyScaledIntoUnchecked(source, inverseNorm, rotated.AsSpan(row * dim, dim), scratch);
        }

        float outerCentroid = 0.0f;
        foreach (float centroid in centroids)
        {
            outerCentroid = Math.Max(outerCentroid, MathF.Abs(centroid));
        }

        double shape = (dim - 1.0) / 2.0;
        double probabilityHigh = Codebook.BetaCdf((outerCentroid + 1.0) / 2.0, shape);
        double probabilityLow = 1.0 - probabilityHigh;
        int lowIndex = (int)(n * probabilityLow);
        int highIndex = Math.Min(n - 1, (int)(n * probabilityHigh));
        highIndex = Math.Max(highIndex, lowIndex + 1);

        float[] shift = new float[dim];
        float[] scale = new float[dim];
        Array.Fill(scale, 1.0f);
        uint[] keys = new uint[n];
        for (int d = 0; d < dim; d++)
        {
            for (int row = 0; row < n; row++)
            {
                keys[row] = ToSortKey(rotated[row * dim + d]);
            }

            float quantileLow = FromSortKey(SelectNth(keys, lowIndex));
            float quantileHigh = FromSortKey(SelectNth(keys, highIndex));
            float quantileSpan = quantileHigh - quantileLow;
            if (quantileSpan > 1e-6f)
            {
                scale[d] = (outerCentroid - -outerCentroid) / quantileSpan;
                shift[d] = -outerCentroid / scale[d] - quantileLow;
            }
        }

        return (shift, scale);
    }

    private static uint SelectNth(uint[] values, int nth)
    {
        int left = 0;
        int right = values.Length - 1;
        int depthLimit = 2 * BitOperations.Log2((uint)Math.Max(values.Length, 1));

        while (left < right)
        {
            if (depthLimit-- == 0)
            {
                Array.Sort(values, left, right - left + 1);
                return values[nth];
            }

            uint pivot = MedianOfThree(values[left], values[left + ((right - left) >> 1)], values[right]);
            int less = left;
            int current = left;
            int greater = right;
            while (current <= greater)
            {
                uint value = values[current];
                if (value < pivot)
                {
                    (values[less], values[current]) = (values[current], values[less]);
                    less++;
                    current++;
                }
                else if (value > pivot)
                {
                    (values[current], values[greater]) = (values[greater], values[current]);
                    greater--;
                }
                else
                {
                    current++;
                }
            }

            if (nth < less)
            {
                right = less - 1;
            }
            else if (nth > greater)
            {
                left = greater + 1;
            }
            else
            {
                return values[nth];
            }
        }

        return values[left];
    }

    private static uint MedianOfThree(uint first, uint middle, uint last)
    {
        if (first > middle)
        {
            (first, middle) = (middle, first);
        }

        if (middle > last)
        {
            (middle, last) = (last, middle);
        }

        if (first > middle)
        {
            middle = first;
        }

        return middle;
    }

    private static uint ToSortKey(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        uint mask = (bits & 0x80000000u) != 0 ? uint.MaxValue : 0x80000000u;
        return bits ^ mask;
    }

    private static float FromSortKey(uint key)
    {
        uint mask = (key & 0x80000000u) != 0 ? 0x80000000u : uint.MaxValue;
        return BitConverter.UInt32BitsToSingle(key ^ mask);
    }
}
