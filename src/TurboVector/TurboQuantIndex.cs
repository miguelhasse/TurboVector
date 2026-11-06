using System.Buffers;
namespace TurboVector;

public class SearchResults(float[] scores, long[] indices, int nq, int k)
{
    /// <summary>Singleton empty result — avoids allocation on the zero-dim fast path.</summary>
    public static readonly SearchResults Empty = new([], [], 0, 0);
    public float[] Scores { get; } = scores;
    public long[] Indices { get; } = indices;
    public int Nq { get; } = nq;
    public int K { get; } = k;

    public ReadOnlySpan<float> ScoresForQuery(int qi)
    {
        if ((uint)qi >= (uint)Nq)
        {
            throw new ArgumentOutOfRangeException(nameof(qi), $"Query index {qi} is out of range [0, {Nq}).");
        }

        return Scores.AsSpan(qi * K, K);
    }

    public ReadOnlySpan<long> IndicesForQuery(int qi)
    {
        if ((uint)qi >= (uint)Nq)
        {
            throw new ArgumentOutOfRangeException(nameof(qi), $"Query index {qi} is out of range [0, {Nq}).");
        }

        return Indices.AsSpan(qi * K, K);
    }
}

/// <summary>
/// TurboQuant vector index: compresses high-dimensional vectors to 2-4 bits
/// per coordinate with near-optimal distortion. Data-oblivious — no training required.
/// </summary>
public sealed class TurboQuantIndex
{
    private sealed class BlockedCache
    {
        public byte[] Data { get; }
        public int NBlocks { get; }
        public byte[]? SubCodes { get; }
        public byte[]? Plane2 { get; }

        public BlockedCache(byte[] data, int nBlocks)
        {
            Data = data;
            NBlocks = nBlocks;
        }

        public BlockedCache(byte[] subCodes, byte[] plane2, int nBlocks)
        {
            Data = [];
            SubCodes = subCodes;
            Plane2 = plane2;
            NBlocks = nBlocks;
        }
    }

    private int? _dim;
    private readonly int _bitWidth;
    private int _nVectors;
    private byte[] _packedCodes;
    private int _packedLength;   // actual bytes used in _packedCodes (capacity may be larger)
    private float[] _scales;
    private Lazy<float[]>? _rotation;
    private Lazy<(float[] Boundaries, float[] Centroids)>? _codebook;
    // volatile: reassigned (or nulled) on every Add/SwapRemove; volatile ensures
    // any concurrent reader sees the updated reference after a write.
    // Note: concurrent Add/SwapRemove with Search still requires external locking
    // to prevent inconsistencies between _nVectors, _packedCodes, _scales, and _cachedBlocked.
    private volatile BlockedCache? _cachedBlocked;
    private readonly Lock _blockCacheLock = new();

    public TurboQuantIndex(int dim, int bitWidth)
    {
        ValidateBitWidth(bitWidth);
        ValidateDim(dim);

        _dim = dim;
        _bitWidth = bitWidth;
        _packedCodes = [];
        _packedLength = 0;
        _scales = [];
        InitializeLazies(dim);
    }

    private TurboQuantIndex(int? dim, int bitWidth, int nVectors, byte[] packedCodes, float[] scales)
    {
        ValidateBitWidth(bitWidth);

        if (dim is not null)
        {
            ValidateDim(dim.Value);
        }

        ArgumentOutOfRangeException.ThrowIfNegative(nVectors);
        ArgumentNullException.ThrowIfNull(packedCodes);
        ArgumentNullException.ThrowIfNull(scales);

        _dim = dim;
        _bitWidth = bitWidth;
        _nVectors = nVectors;
        _packedCodes = packedCodes;
        _packedLength = packedCodes.Length;
        _scales = scales;

        if (dim is not null)
        {
            int expectedPackedLength = checked(dim.Value / 8 * bitWidth * nVectors);
            if (packedCodes.Length != expectedPackedLength)
            {
                throw new ArgumentException($"Expected packedCodes length {expectedPackedLength}, got {packedCodes.Length}.", nameof(packedCodes));
            }
        }
        else if (packedCodes.Length != 0)
        {
            throw new ArgumentException("packedCodes must be empty when dim is not set.", nameof(packedCodes));
        }
        else if (nVectors != 0)
        {
            throw new ArgumentException("nVectors must be zero when dim is not set.", nameof(nVectors));
        }

        if (scales.Length != nVectors)
        {
            throw new ArgumentException($"Expected scales length {nVectors}, got {scales.Length}.", nameof(scales));
        }

        if (dim is not null)
        {
            InitializeLazies(dim.Value);
        }
    }

    public static TurboQuantIndex NewLazy(int bitWidth)
    {
        ValidateBitWidth(bitWidth);
        return new TurboQuantIndex(null, bitWidth, 0, [], []);
    }

    public void Add(ReadOnlySpan<float> vectors)
    {
        int dim = _dim ?? throw new InvalidOperationException("dim is not set. Use Add2D on a lazy index to commit the dimension first.");
        EnsureInitialized();

        if (vectors.Length % dim != 0)
        {
            throw new ArgumentException("vectors length must be divisible by dim.", nameof(vectors));
        }

        int n = vectors.Length / dim;
        var rotation = _rotation!.Value;
        var (boundaries, centroids) = _codebook!.Value;

        int batchPackedLen = checked(n * (dim / 8) * _bitWidth);
        int newPackedLength = _packedLength + batchPackedLen;
        if (newPackedLength > _packedCodes.Length)
        {
            int newCapacity = Math.Max(newPackedLength, Math.Max(batchPackedLen, _packedCodes.Length * 2));
            var newBuf = new byte[newCapacity];
            _packedCodes.AsSpan(0, _packedLength).CopyTo(newBuf);
            _packedCodes = newBuf;
        }

        int newScalesLen = _nVectors + n;
        if (newScalesLen > _scales.Length)
        {
            int newCapacity = Math.Max(newScalesLen, Math.Max(n, _scales.Length * 2));
            var newScales = new float[newCapacity];
            _scales.AsSpan(0, _nVectors).CopyTo(newScales);
            _scales = newScales;
        }

        Encoder.EncodeInto(
            vectors, n, dim, rotation, boundaries, centroids, _bitWidth,
            _packedCodes.AsSpan(_packedLength, batchPackedLen),
            _scales.AsSpan(_nVectors, n));

        _packedLength = newPackedLength;
        _nVectors += n;
        _cachedBlocked = null;
    }

    public void Add2D(ReadOnlySpan<float> vectors, int dim)
    {
        if (_dim is null)
        {
            ValidateDim(dim);
            _dim = dim;
            InitializeLazies(dim);
        }
        else if (_dim.Value != dim)
        {
            throw new ArgumentException($"dim mismatch: index dimension {_dim.Value} does not match provided dim {dim}.", nameof(dim));
        }

        Add(vectors);
    }

    public SearchResults Search(ReadOnlySpan<float> queries, int k)
        => SearchWithMask(queries, k, default);

    public SearchResults SearchWithMask(ReadOnlySpan<float> queries, int k, ReadOnlySpan<bool> mask)
    {
        if (_dim is null)
        {
            return SearchResults.Empty;
        }

        int dim = _dim.Value;
        if (queries.Length % dim != 0)
        {
            throw new ArgumentException("queries length must be divisible by dim.", nameof(queries));
        }

        ArgumentOutOfRangeException.ThrowIfNegative(k);

        if (!mask.IsEmpty && mask.Length != _nVectors)
        {
            throw new ArgumentException($"mask length {mask.Length} does not match index length {_nVectors}.", nameof(mask));
        }

        int nq = queries.Length / dim;
        var rotation = _rotation!.Value;
        var (_, centroids) = _codebook!.Value;
        var blocked = EnsureBlocked();

        // Single-pass: pack the bool mask AND count allowed vectors simultaneously.
        // If no mask, skip entirely. Rent from ArrayPool to avoid per-search allocation.
        ulong[]? rentedMask = null;
        int wordCount = 0;
        int nAllowed;
        if (mask.IsEmpty || _nVectors == 0)
        {
            nAllowed = _nVectors;
        }
        else
        {
            wordCount = (_nVectors + 63) >> 6;
            rentedMask = ArrayPool<ulong>.Shared.Rent(wordCount);
            rentedMask.AsSpan(0, wordCount).Clear();
            int cnt = 0;
            for (int i = 0; i < _nVectors; i++)
            {
                if (mask[i]) { rentedMask[i >> 6] |= 1UL << (i & 63); cnt++; }
            }
            nAllowed = cnt;
        }

        int effectiveK = Math.Min(k, Math.Min(_nVectors, nAllowed));
        SearchResults result;
        try
        {
            if (_bitWidth == 3)
            {
                var (scores, indices) = global::TurboVector.Search.Search3Bit_(
                    queries,
                    nq,
                    rotation,
                    blocked.SubCodes!,
                    blocked.Plane2!,
                    centroids,
                    _scales,
                    dim,
                    _nVectors,
                    blocked.NBlocks,
                    effectiveK,
                    rentedMask,
                    wordCount);
                result = new SearchResults(scores, indices, nq, effectiveK);
            }
            else
            {
                var (Scores, Indices) = global::TurboVector.Search.Search_(
                    queries,
                    nq,
                    rotation,
                    blocked.Data,
                    centroids,
                    _scales,
                    _bitWidth,
                    dim,
                    _nVectors,
                    blocked.NBlocks,
                    effectiveK,
                    rentedMask,
                    wordCount);
                result = new SearchResults(Scores, Indices, nq, effectiveK);
            }
        }
        finally
        {
            if (rentedMask is not null) ArrayPool<ulong>.Shared.Return(rentedMask);
        }

        return result;
    }

    public void Prepare()
    {
        if (_dim is null)
        {
            return;
        }

        _ = _rotation!.Value;
        _ = _codebook!.Value;
        _ = EnsureBlocked();
    }

    public void Write(string path)
        => Io.Write(path, _bitWidth, _dim ?? 0, _nVectors,
            new ReadOnlySpan<byte>(_packedCodes, 0, _packedLength),
            new ReadOnlySpan<float>(_scales, 0, _nVectors));

    public static TurboQuantIndex Load(string path)
    {
        var (bitWidth, dim, nVectors, packedCodes, scales) = Io.Load(path);
        int? dimOpt = dim == 0 ? null : dim;
        return FromParts(dimOpt, bitWidth, nVectors, packedCodes, scales);
    }

    internal static TurboQuantIndex FromParts(int? dim, int bitWidth, int nVectors, byte[] packedCodes, float[] scales)
        => new(dim, bitWidth, nVectors, packedCodes, scales);

    public int SwapRemove(int idx)
    {
        if (_dim is null)
        {
            throw new InvalidOperationException("Dimension is not set.");
        }

        if ((uint)idx >= (uint)_nVectors)
        {
            throw new ArgumentOutOfRangeException(nameof(idx), $"idx {idx} is out of bounds for index of length {_nVectors}.");
        }

        int dim = _dim.Value;
        int bytesPerVec = checked(dim * _bitWidth / 8);
        int last = _nVectors - 1;

        if (idx != last)
        {
            Array.Copy(_packedCodes, last * bytesPerVec, _packedCodes, idx * bytesPerVec, bytesPerVec);
            _scales[idx] = _scales[last];
        }

        // Shrink logical length without reallocating — capacity stays for future Adds.
        _packedLength -= bytesPerVec;
        _nVectors--;
        _cachedBlocked = null;
        return last;
    }

    public int Len => _nVectors;

    public bool IsEmpty => _nVectors == 0;

    public int Dim => _dim ?? 0;

    public int? DimOpt => _dim;

    public int BitWidth => _bitWidth;

    internal ReadOnlySpan<byte> PackedCodes => new(_packedCodes, 0, _packedLength);

    internal ReadOnlySpan<float> Scales => new(_scales, 0, _nVectors);

    private void InitializeLazies(int dim)
    {
        _rotation = MakeRotationLazy(dim);
        _codebook = MakeCodebookLazy(_bitWidth, dim);
        _cachedBlocked = null;
    }

    private void EnsureInitialized()
    {
        if (_rotation is null || _codebook is null)
        {
            throw new InvalidOperationException("Index is not initialized.");
        }
    }

    private static Lazy<float[]> MakeRotationLazy(int dim) =>
        new(() => Rotation.MakeRotationMatrix(dim), LazyThreadSafetyMode.ExecutionAndPublication);

    private static Lazy<(float[] Boundaries, float[] Centroids)> MakeCodebookLazy(int bitWidth, int dim) =>
        new(() => Codebook.Compute(bitWidth, dim), LazyThreadSafetyMode.ExecutionAndPublication);

    private BlockedCache EnsureBlocked()
    {
        var cached = _cachedBlocked;
        if (cached is not null)
        {
            return cached;
        }

        lock (_blockCacheLock)
        {
            cached = _cachedBlocked;
            if (cached is not null) return cached;
            _cachedBlocked = cached = BuildBlockedCache();
            return cached;
        }
    }

    private BlockedCache BuildBlockedCache()
    {
        int dim = _dim!.Value;
        var codes = new ReadOnlySpan<byte>(_packedCodes, 0, _packedLength);
        if (_bitWidth == 3)
        {
            var (subCodes, plane2, nBlocks) = Pack.Repack3Bit(codes, _nVectors, dim);
            return new BlockedCache(subCodes, plane2, nBlocks);
        }

        var (blocked, nBlocksNibble) = Pack.Repack(codes, _nVectors, _bitWidth, dim);
        return new BlockedCache(blocked, nBlocksNibble);
    }

    private static void ValidateBitWidth(int bitWidth)
    {
        if (bitWidth < 2 || bitWidth > 4)
        {
            throw new ArgumentOutOfRangeException(nameof(bitWidth), "bitWidth must be between 2 and 4.");
        }
    }

    private static void ValidateDim(int dim)
    {
        if (dim <= 0 || dim % 8 != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dim), "dim must be positive and divisible by 8.");
        }
    }
}
