using System.Buffers;

namespace TurboVector;

/// <summary>
/// Random orthogonal rotation matrix generation.
/// Generates a deterministic orthogonal matrix via QR decomposition of
/// a seeded Gaussian random matrix.
/// </summary>
public static class Rotation
{
    /// <summary>
    /// Generate a dim x dim orthogonal matrix (deterministic, seeded).
    /// Returns a row-major flat array of length dim * dim.
    /// </summary>
    public static float[] MakeRotationMatrix(int dim)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dim);

        int matrixSize = checked(dim * dim);
        double[] a = ArrayPool<double>.Shared.Rent(matrixSize);
        try
        {
            var rng = new ChaCha8Rng(Constants.RotationSeed);
            for (int i = 0; i < matrixSize; i++)
            {
                a[i] = rng.NextStandardNormal();
            }

            double[] q = HouseholderQR(a, dim);
            try
            {
                float[] rotation = new float[matrixSize];
                for (int i = 0; i < dim; i++)
                {
                    for (int j = 0; j < dim; j++)
                    {
                        rotation[(i * dim) + j] = (float)q[(j * dim) + i];
                    }
                }

                return rotation;
            }
            finally
            {
                ArrayPool<double>.Shared.Return(q);
            }
        }
        finally
        {
            ArrayPool<double>.Shared.Return(a);
        }
    }

    private static double[] HouseholderQR(double[] a, int dim)
    {
        int matrixSize = checked(dim * dim);
        double[]? q = ArrayPool<double>.Shared.Rent(matrixSize);
        double[]? v = null;
        bool success = false;
        try
        {
            Array.Clear(q, 0, matrixSize);
            for (int i = 0; i < dim; i++)
            {
                q[(i * dim) + i] = 1.0;
            }

            v = ArrayPool<double>.Shared.Rent(dim);

            for (int k = 0; k < dim; k++)
            {
                double norm = 0.0;
                for (int i = k; i < dim; i++)
                {
                    v[i] = a[(k * dim) + i];
                    norm += v[i] * v[i];
                }

                norm = Math.Sqrt(norm);
                if (norm < 1e-14)
                {
                    continue;
                }

                double sign = v[k] >= 0.0 ? 1.0 : -1.0;
                v[k] += sign * norm;

                double vv = 0.0;
                for (int i = k; i < dim; i++)
                {
                    vv += v[i] * v[i];
                }

                if (vv < 1e-28)
                {
                    continue;
                }

                double beta = 2.0 / vv;

                for (int j = k; j < dim; j++)
                {
                    double dot = 0.0;
                    for (int i = k; i < dim; i++)
                    {
                        dot += v[i] * a[(j * dim) + i];
                    }

                    dot *= beta;
                    for (int i = k; i < dim; i++)
                    {
                        a[(j * dim) + i] -= dot * v[i];
                    }
                }

                for (int row = 0; row < dim; row++)
                {
                    double dot = 0.0;
                    for (int i = k; i < dim; i++)
                    {
                        dot += q[(i * dim) + row] * v[i];
                    }

                    dot *= beta;
                    for (int i = k; i < dim; i++)
                    {
                        q[(i * dim) + row] -= dot * v[i];
                    }
                }
            }

            for (int k = 0; k < dim; k++)
            {
                if (a[(k * dim) + k] < 0.0)
                {
                    for (int row = 0; row < dim; row++)
                    {
                        q[(k * dim) + row] = -q[(k * dim) + row];
                    }
                }
            }

            success = true;
            double[] result = q;
            q = null;
            return result;
        }
        finally
        {
            if (v != null) ArrayPool<double>.Shared.Return(v);
            if (!success && q != null) ArrayPool<double>.Shared.Return(q);
        }
    }
}

internal sealed class ChaCha8Rng
{
    private static readonly uint[] Constants =
    [
        0x61707865,
        0x3320646e,
        0x79622d32,
        0x6b206574,
    ];

    private readonly uint[] _state = new uint[16];
    private readonly uint[] _block = new uint[16];
    private int _index = 16;
    private bool _hasSpare;
    private double _spare;

    public ChaCha8Rng(ulong seed)
    {
        _state[0] = Constants[0];
        _state[1] = Constants[1];
        _state[2] = Constants[2];
        _state[3] = Constants[3];
        _state[4] = (uint)(seed & 0xffff_ffffUL);
        _state[5] = (uint)(seed >> 32);
        _state[6] = 0;
        _state[7] = 0;
        _state[8] = 0;
        _state[9] = 0;
        _state[10] = 0;
        _state[11] = 0;
        _state[12] = 0;
        _state[13] = 0;
        _state[14] = 0;
        _state[15] = 0;
    }

    public double NextStandardNormal()
    {
        if (_hasSpare)
        {
            _hasSpare = false;
            return _spare;
        }

        double u;
        double v;
        double s;
        do
        {
            u = (2.0 * NextDouble()) - 1.0;
            v = (2.0 * NextDouble()) - 1.0;
            s = (u * u) + (v * v);
        }
        while (s >= 1.0 || s == 0.0);

        double mul = Math.Sqrt((-2.0 * Math.Log(s)) / s);
        _spare = v * mul;
        _hasSpare = true;
        return u * mul;
    }

    private double NextDouble()
    {
        ulong bits = ((ulong)NextUInt32() << 21) | ((ulong)NextUInt32() >> 11);
        return bits * (1.0 / (1UL << 53));
    }

    private uint NextUInt32()
    {
        if (_index >= _block.Length)
        {
            GenerateBlock();
        }

        return _block[_index++];
    }

    private void GenerateBlock()
    {
        Array.Copy(_state, _block, _state.Length);

        for (int i = 0; i < 4; i++)
        {
            QuarterRound(_block, 0, 4, 8, 12);
            QuarterRound(_block, 1, 5, 9, 13);
            QuarterRound(_block, 2, 6, 10, 14);
            QuarterRound(_block, 3, 7, 11, 15);
            QuarterRound(_block, 0, 5, 10, 15);
            QuarterRound(_block, 1, 6, 11, 12);
            QuarterRound(_block, 2, 7, 8, 13);
            QuarterRound(_block, 3, 4, 9, 14);
        }

        for (int i = 0; i < _block.Length; i++)
        {
            _block[i] += _state[i];
        }

        _state[12]++;
        if (_state[12] == 0)
        {
            _state[13]++;
        }

        _index = 0;
    }

    private static void QuarterRound(uint[] state, int a, int b, int c, int d)
    {
        state[a] += state[b];
        state[d] = RotateLeft(state[d] ^ state[a], 16);
        state[c] += state[d];
        state[b] = RotateLeft(state[b] ^ state[c], 12);
        state[a] += state[b];
        state[d] = RotateLeft(state[d] ^ state[a], 8);
        state[c] += state[d];
        state[b] = RotateLeft(state[b] ^ state[c], 7);
    }

    private static uint RotateLeft(uint value, int amount)
        => (value << amount) | (value >> (32 - amount));
}
