using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace TurboVector;

/// <summary>
/// Deterministic two-round globally-permuted block-Hadamard rotation.
/// </summary>
public sealed class Rotation
{
    private const int Rounds = 2;
    private static readonly byte[] Seed =
    [
        164, 143, 161, 123, 88, 50, 61, 10, 234, 184, 161, 204, 105, 1, 20, 184,
        43, 140, 200, 117, 24, 180, 247, 84, 141, 68, 110, 161, 228, 223, 32, 242,
    ];

    private readonly int _dim;
    private readonly int _blockSize;
    private readonly float _inverseSqrtBlock;
    private readonly int[][] _permutations;
    private readonly float[][] _signs;

    internal Rotation(int dim)
    {
        InputValidation.ValidateDimension(dim, nameof(dim));
        _dim = dim;
        _blockSize = dim & -dim;
        _inverseSqrtBlock = 1.0f / MathF.Sqrt(_blockSize);

        _permutations = new int[Rounds][];
        _signs = new float[Rounds][];
        var rng = new ChaCha8Rng(Seed);
        for (int round = 0; round < Rounds; round++)
        {
            float[] signs = new float[dim];
            for (int i = 0; i < dim; i++)
            {
                signs[i] = (rng.NextUInt32() & 1) != 0 ? -1.0f : 1.0f;
            }

            int[] permutation = new int[dim];
            for (int i = 0; i < dim; i++)
            {
                permutation[i] = i;
            }

            for (int i = dim - 1; i > 0; i--)
            {
                int j = (int)(rng.NextUInt64() % (uint)(i + 1));
                (permutation[i], permutation[j]) = (permutation[j], permutation[i]);
            }

            _signs[round] = signs;
            _permutations[round] = permutation;
        }
    }

    /// <summary>
    /// Create a rotation for the specified positive dimension, which must be
    /// a multiple of eight and no greater than <see cref="Constants.MaxDim"/>.
    /// </summary>
    public static Rotation Create(int dim) => new(dim);

    /// <summary>The vector dimensionality this rotation is built for.</summary>
    public int Dim => _dim;

    /// <summary>Rotate one vector in place, using scratch that does not overlap the vector.</summary>
    public void Apply(Span<float> row, Span<float> scratch)
    {
        ValidateBuffers(row, scratch);
        ApplyRound(row, scratch, 0, 1.0f, scaled: false);
        ApplyRound(scratch, row, 1, 1.0f, scaled: false);
    }

    /// <summary>
    /// Rotate a normalized vector directly into a destination. The source is
    /// left unchanged unless it overlaps the destination. Scratch must not
    /// overlap either span. The input scaling is fused into the first gather.
    /// </summary>
    public void ApplyScaledInto(
        ReadOnlySpan<float> source,
        float inverseNorm,
        Span<float> destination,
        Span<float> scratch)
    {
        if (source.Length != _dim || destination.Length != _dim || scratch.Length != _dim)
        {
            throw new ArgumentException("Source, destination, and scratch lengths must equal the rotation dimension.");
        }

        if (source.Overlaps(scratch) || destination.Overlaps(scratch))
        {
            throw new ArgumentException("Scratch must not overlap the source or destination.", nameof(scratch));
        }

        ApplyScaledIntoUnchecked(source, inverseNorm, destination, scratch);
    }

    internal void ApplyScaledIntoUnchecked(
        ReadOnlySpan<float> source,
        float inverseNorm,
        Span<float> destination,
        Span<float> scratch)
    {
        ApplyRound(source, scratch, 0, inverseNorm, scaled: true);
        ApplyRound(scratch, destination, 1, 1.0f, scaled: false);
    }

    /// <summary>
    /// Materialize the row-major dense matrix for callers that require one.
    /// The index and encoder use <see cref="Apply"/> directly and do not
    /// allocate this quadratic representation.
    /// </summary>
    public static float[] MakeRotationMatrix(int dim)
    {
        var rotation = new Rotation(dim);
        int size = checked(dim * dim);
        float[] matrix = new float[size];
        float[] column = new float[dim];
        float[] scratch = new float[dim];

        for (int j = 0; j < dim; j++)
        {
            Array.Clear(column);
            column[j] = 1.0f;
            rotation.Apply(column, scratch);
            for (int i = 0; i < dim; i++)
            {
                matrix[(i * dim) + j] = column[i];
            }
        }

        return matrix;
    }

    private void ValidateBuffers(Span<float> row, Span<float> scratch)
    {
        if (row.Length != _dim || scratch.Length != _dim)
        {
            throw new ArgumentException("Row and scratch lengths must equal the rotation dimension.");
        }

        if (row.Overlaps(scratch))
        {
            throw new ArgumentException("Scratch must not overlap the row.", nameof(scratch));
        }
    }

    private void ApplyRound(
        ReadOnlySpan<float> source,
        Span<float> destination,
        int round,
        float inverseNorm,
        bool scaled)
    {
        int[] permutation = _permutations[round];
        float[] signs = _signs[round];
        for (int i = 0; i < _dim; i++)
        {
            float value = source[permutation[i]];
            if (scaled)
            {
                value *= inverseNorm;
            }

            destination[i] = value * signs[i];
        }

        for (int offset = 0; offset < _dim; offset += _blockSize)
        {
            ApplyHadamard(destination.Slice(offset, _blockSize));
        }
    }

    private void ApplyHadamard(Span<float> block)
    {
        int length = 1;
        bool useAvx2 = Avx.IsSupported && block.Length >= 16;
        while (length < block.Length)
        {
            int groupSize = length << 1;
            for (int group = 0; group < block.Length; group += groupSize)
            {
                int j = 0;
                if (useAvx2 && length >= Vector256<float>.Count)
                {
                    unsafe
                    {
                        fixed (float* p = block)
                        {
                            for (; j <= length - Vector256<float>.Count; j += Vector256<float>.Count)
                            {
                                Vector256<float> a = Avx.LoadVector256(p + group + j);
                                Vector256<float> b = Avx.LoadVector256(p + group + length + j);
                                Avx.Store(p + group + j, Avx.Add(a, b));
                                Avx.Store(p + group + length + j, Avx.Subtract(a, b));
                            }
                        }
                    }
                }

                for (; j < length; j++)
                {
                    float a = block[group + j];
                    float b = block[group + length + j];
                    block[group + j] = a + b;
                    block[group + length + j] = a - b;
                }
            }

            length = groupSize;
        }

        int i = 0;
        if (useAvx2)
        {
            unsafe
            {
                fixed (float* p = block)
                {
                    Vector256<float> scale = Vector256.Create(_inverseSqrtBlock);
                    for (; i <= block.Length - Vector256<float>.Count; i += Vector256<float>.Count)
                    {
                        Avx.Store(p + i, Avx.Multiply(Avx.LoadVector256(p + i), scale));
                    }
                }
            }
        }

        for (; i < block.Length; i++)
        {
            block[i] *= _inverseSqrtBlock;
        }
    }
}

internal sealed class ChaCha8Rng
{
    private static ReadOnlySpan<uint> Constants =>
    [
        0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
    ];

    private readonly uint[] _state = new uint[16];
    private readonly uint[] _block = new uint[16];
    private int _index = 16;

    public ChaCha8Rng(ReadOnlySpan<byte> seed)
    {
        if (seed.Length != 32)
        {
            throw new ArgumentException("ChaCha8 seed must be exactly 32 bytes.", nameof(seed));
        }

        Constants.CopyTo(_state);
        for (int i = 0; i < 8; i++)
        {
            _state[4 + i] = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(seed.Slice(i * 4, 4));
        }
    }

    public uint NextUInt32()
    {
        if (_index == _block.Length)
        {
            GenerateBlock();
        }

        return _block[_index++];
    }

    public ulong NextUInt64()
    {
        uint low = NextUInt32();
        uint high = NextUInt32();
        return low | ((ulong)high << 32);
    }

    private void GenerateBlock()
    {
        _state.CopyTo(_block, 0);
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

        if (++_state[12] == 0)
        {
            _state[13]++;
        }

        _index = 0;
    }

    private static void QuarterRound(uint[] state, int a, int b, int c, int d)
    {
        state[a] += state[b];
        state[d] = BitOperations.RotateLeft(state[d] ^ state[a], 16);
        state[c] += state[d];
        state[b] = BitOperations.RotateLeft(state[b] ^ state[c], 12);
        state[a] += state[b];
        state[d] = BitOperations.RotateLeft(state[d] ^ state[a], 8);
        state[c] += state[d];
        state[b] = BitOperations.RotateLeft(state[b] ^ state[c], 7);
    }
}
