namespace TurboVector.Benchmarks;

internal static class BenchmarkData
{
    public static float[] MakeVectors(int n, int dim, ulong seed = 42)
    {
        ulong state = unchecked(seed * 0x9E3779B97F4A7C15UL);
        float[] data = new float[n * dim];
        for (int i = 0; i < data.Length; i++)
        {
            state = unchecked(state * 6364136223846793005UL + 1442695040888963407UL);
            uint bits = ((uint)(state >> 32) & 0x007FFFFFu) | 0x3F800000u;
            data[i] = BitConverter.Int32BitsToSingle((int)bits) - 1.0f;
            data[i] = data[i] * 2.0f - 1.0f;
        }

        return data;
    }

    public static bool[] MakeMask(int n, double allowFraction)
    {
        int allowed = Math.Clamp((int)(n * allowFraction), 0, n);
        bool[] mask = new bool[n];
        for (int i = 0; i < allowed; i++)
        {
            mask[i] = true;
        }

        return mask;
    }

    public static ulong[] MakeAllowlist(int n, double allowFraction)
    {
        int allowed = Math.Clamp((int)(n * allowFraction), 0, n);
        ulong[] allowlist = new ulong[allowed];
        for (int i = 0; i < allowed; i++)
        {
            allowlist[i] = (ulong)i;
        }

        return allowlist;
    }
}
