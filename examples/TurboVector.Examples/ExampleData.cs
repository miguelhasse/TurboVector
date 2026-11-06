using System.Numerics;

internal static class ExampleData
{
    private const ulong LcgMultiplier = 6_364_136_223_846_793_005UL;
    private const ulong LcgIncrement = 1_442_695_040_888_963_407UL;

    public static float[] CreateLcgVectors(int count, int dim, ulong seed)
    {
        float[] vectors = new float[checked(count * dim)];
        ulong state = seed;
        for (int i = 0; i < vectors.Length; i++)
        {
            state = unchecked(state * LcgMultiplier + LcgIncrement);
            vectors[i] = ((state >> 33) / (float)(1U << 31)) - 0.5f;
        }

        return vectors;
    }

    public static float[] CreateNormalizedUniformVectors(int count, int dim, ulong seed)
    {
        float[] vectors = new float[checked(count * dim)];
        ulong state = seed | 1UL;
        for (int i = 0; i < vectors.Length; i++)
        {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            vectors[i] = (float)((double)state / ulong.MaxValue) - 0.5f;
        }

        NormalizeRows(vectors, dim);
        return vectors;
    }

    public static void NormalizeRows(Span<float> values, int dim)
    {
        for (int offset = 0; offset < values.Length; offset += dim)
        {
            Span<float> row = values.Slice(offset, dim);
            double sum = 0.0;
            for (int i = 0; i < row.Length; i++)
            {
                sum += row[i] * row[i];
            }

            if (sum <= 1e-20)
            {
                continue;
            }

            float inverseNorm = (float)(1.0 / Math.Sqrt(sum));
            for (int i = 0; i < row.Length; i++)
            {
                row[i] *= inverseNorm;
            }
        }
    }

    public static void FillNormal(Span<float> values, ulong seed)
    {
        var random = new Xoshiro256StarStar(seed);
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = random.NextNormal();
        }
    }

    private sealed class Xoshiro256StarStar
    {
        private readonly ulong[] _state = new ulong[4];

        public Xoshiro256StarStar(ulong seed)
        {
            ulong state = seed;
            for (int i = 0; i < _state.Length; i++)
            {
                state = unchecked(state + 0x9E37_79B9_7F4A_7C15UL);
                ulong value = state;
                value = unchecked((value ^ (value >> 30)) * 0xBF58_476D_1CE4_E5B9UL);
                value = unchecked((value ^ (value >> 27)) * 0x94D0_49BB_1331_11EBUL);
                _state[i] = value ^ (value >> 31);
            }
        }

        private ulong Next()
        {
            ulong result = unchecked(BitOperations.RotateLeft(_state[1] * 5, 7) * 9);
            ulong temporary = _state[1] << 17;
            _state[2] ^= _state[0];
            _state[3] ^= _state[1];
            _state[1] ^= _state[2];
            _state[0] ^= _state[3];
            _state[2] ^= temporary;
            _state[3] = BitOperations.RotateLeft(_state[3], 45);
            return result;
        }

        public float NextNormal()
        {
            float u1 = Math.Max((Next() >> 40) / (float)(1U << 24), 1.17549435E-38f);
            float u2 = (Next() >> 40) / (float)(1U << 24);
            float radius = MathF.Sqrt(-2.0f * MathF.Log(u1));
            float theta = 2.0f * MathF.PI * u2;
            return radius * MathF.Cos(theta);
        }
    }
}
