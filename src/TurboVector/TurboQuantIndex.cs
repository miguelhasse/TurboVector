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
    private float[] _tqplusShift = [];
    private float[] _tqplusScale = [];
    private Lazy<Rotation>? _rotation;
    private Lazy<(float[] Boundaries, float[] Centroids)>? _codebook;
    // volatile: reassigned (or nulled) on every Add/SwapRemove; volatile ensures
    // any concurrent reader sees the updated reference after a write.
    // Note: concurrent Add/SwapRemove with Search still requires external locking
    // to prevent inconsistencies between _nVectors, _packedCodes, _scales, and _cachedBlocked.
    private volatile BlockedCache? _cachedBlocked;
    private readonly Lock _blockCacheLock = new();
    private readonly Lock _syncLock = new();
    private string? _syncPath;
    private SyncCursor? _syncCursor;
    private bool _syncAppendOnly;
    private bool _syncAppendCompatible;

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

    private TurboQuantIndex(
        int? dim,
        int bitWidth,
        int nVectors,
        byte[] packedCodes,
        float[] scales,
        float[]? tqPlusShift = null,
        float[]? tqPlusScale = null)
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
        _tqplusShift = tqPlusShift ?? [];
        _tqplusScale = tqPlusScale ?? [];

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

        if (_tqplusShift.Length != _tqplusScale.Length
            || (_tqplusShift.Length != 0 && _tqplusShift.Length != dim))
        {
            throw new ArgumentException("TQ+ calibration arrays must both be empty or have one value per dimension.");
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
        AddCore(vectors, dim, _rotation!, _codebook!);
    }

    private void AddCore(
        ReadOnlySpan<float> vectors,
        int dim,
        Lazy<Rotation> rotationLazy,
        Lazy<(float[] Boundaries, float[] Centroids)> codebookLazy)
    {
        if (vectors.Length % dim != 0)
        {
            throw new ArgumentException("vectors length must be divisible by dim.", nameof(vectors));
        }

        int n = vectors.Length / dim;
        var rotation = rotationLazy.Value;
        var (boundaries, centroids) = codebookLazy.Value;

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

        Encoder.EncodeIntoCalibrated(
            vectors, n, dim, rotation, boundaries, centroids, _tqplusShift, _tqplusScale, _bitWidth,
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
            Lazy<Rotation> rotation = MakeRotationLazy(dim);
            Lazy<(float[] Boundaries, float[] Centroids)> codebook = MakeCodebookLazy(_bitWidth, dim);
            AddCore(vectors, dim, rotation, codebook);
            _dim = dim;
            _rotation = rotation;
            _codebook = codebook;
            return;
        }
        else if (_dim.Value != dim)
        {
            throw new ArgumentException($"dim mismatch: index dimension {_dim.Value} does not match provided dim {dim}.", nameof(dim));
        }

        Add(vectors);
    }

    public void Calibrate(ReadOnlySpan<float> sample)
    {
        int dim = _dim ?? throw new InvalidOperationException(
            "Dimension is not set. Use Calibrate2D on a lazy index.");
        Calibrate2D(sample, dim);
    }

    public void Calibrate2D(ReadOnlySpan<float> sample, int dim)
    {
        ValidateDim(dim);
        if (_dim is int existingDim && existingDim != dim)
        {
            throw new ArgumentException($"Index dimension {existingDim} does not match provided dim {dim}.", nameof(dim));
        }

        if (sample.Length % dim != 0)
        {
            throw new ArgumentException("sample length must be divisible by dim.", nameof(sample));
        }

        int rows = sample.Length / dim;
        if (rows < 2)
        {
            throw new ArgumentException("Calibration sample must contain at least two vectors.", nameof(sample));
        }

        InputValidation.ValidateValues(sample, dim, nameof(sample));
        Rotation rotation = _rotation?.Value ?? Rotation.Create(dim);
        var (boundaries, centroids) = _codebook?.Value ?? Codebook.Compute(_bitWidth, dim);
        var (shift, scale) = TqPlusCalibration.Fit(sample, rows, dim, rotation, centroids);
        if (shift.All(static value => value == 0.0f) && scale.All(static value => value == 1.0f))
        {
            throw new ArgumentException("Calibration sample is degenerate; no per-coordinate calibration could be fitted.", nameof(sample));
        }

        byte[]? reencoded = null;
        float[]? reencodedScales = null;
        if (_nVectors > 0)
        {
            reencoded = new byte[_packedLength];
            reencodedScales = new float[_nVectors];
            float[] inverseOldScale = new float[dim];
            for (int d = 0; d < dim; d++)
            {
                inverseOldScale[d] = _tqplusScale.Length == 0 ? 1.0f : 1.0f / _tqplusScale[d];
            }

            const int MaxChunkRows = 4096;
            const int MaxChunkCoordinates = 2 * 1024 * 1024;
            int chunkRows = Math.Min(_nVectors, Math.Min(MaxChunkRows, Math.Max(1, MaxChunkCoordinates / dim)));
            int chunkCoordinates = checked(chunkRows * dim);
            int bytesPerRow = checked(dim / 8 * _bitWidth);
            float[]? reconstructed = null;
            float[]? norms = null;
            byte[]? codes = null;
            try
            {
                reconstructed = ArrayPool<float>.Shared.Rent(chunkCoordinates);
                norms = ArrayPool<float>.Shared.Rent(chunkRows);
                codes = ArrayPool<byte>.Shared.Rent(chunkCoordinates);
                for (int start = 0; start < _nVectors; start += chunkRows)
                {
                    int count = Math.Min(chunkRows, _nVectors - start);
                    int coordinates = checked(count * dim);
                    int packedOffset = checked(start * bytesPerRow);
                    int packedLength = checked(count * bytesPerRow);
                    Pack.ExtractCodes(
                        _packedCodes.AsSpan(packedOffset, packedLength),
                        count, _bitWidth, dim, codes.AsSpan(0, coordinates));
                    for (int row = 0; row < count; row++)
                    {
                        double sumSquares = 0.0;
                        int rowStart = row * dim;
                        for (int d = 0; d < dim; d++)
                        {
                            float oldShift = _tqplusShift.Length == 0 ? 0.0f : _tqplusShift[d];
                            float value = centroids[codes[rowStart + d]] * inverseOldScale[d] - oldShift;
                            reconstructed[rowStart + d] = value;
                            sumSquares += (double)value * value;
                        }

                        norms[row] = (float)((double)_scales[start + row] * sumSquares);
                    }

                    Encoder.EncodePrerotatedInto(
                        reconstructed.AsSpan(0, coordinates),
                        norms.AsSpan(0, count),
                        count,
                        dim,
                        boundaries,
                        centroids,
                        shift,
                        scale,
                        _bitWidth,
                        reencoded.AsSpan(packedOffset, packedLength),
                        reencodedScales.AsSpan(start, count));
                }
            }
            finally
            {
                if (codes is not null) ArrayPool<byte>.Shared.Return(codes);
                if (norms is not null) ArrayPool<float>.Shared.Return(norms);
                if (reconstructed is not null) ArrayPool<float>.Shared.Return(reconstructed);
            }
        }

        if (_dim is null)
        {
            _dim = dim;
            _rotation = new Lazy<Rotation>(() => rotation, LazyThreadSafetyMode.ExecutionAndPublication);
            _codebook = new Lazy<(float[] Boundaries, float[] Centroids)>(
                () => (boundaries, centroids),
                LazyThreadSafetyMode.ExecutionAndPublication);
        }

        if (reencoded is not null)
        {
            _packedCodes = reencoded;
            _packedLength = reencoded.Length;
            _scales = reencodedScales!;
            _cachedBlocked = null;
        }

        _tqplusShift = shift;
        _tqplusScale = scale;
        _syncAppendOnly = false;
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
                    wordCount,
                    _tqplusShift,
                    _tqplusScale);
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
                    wordCount,
                    _tqplusShift,
                    _tqplusScale);
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
        => Write(path, Durability.Durable);

    public void Write(string path, Durability durability)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        lock (_syncLock)
        {
            Io.WriteTurboQuantIndex(path, _bitWidth, _dim ?? 0, _nVectors,
                new ReadOnlySpan<byte>(_packedCodes, 0, _packedLength),
                new ReadOnlySpan<float>(_scales, 0, _nVectors), _tqplusShift, _tqplusScale, durability);
            ForgetSyncBindingIfPathMatches(path);
        }
    }

    public byte[] ToBytes()
        => Io.ToBytes(_bitWidth, _dim ?? 0, _nVectors,
            new ReadOnlySpan<byte>(_packedCodes, 0, _packedLength),
            new ReadOnlySpan<float>(_scales, 0, _nVectors), _tqplusShift, _tqplusScale);

    public void WriteTo(Stream output)
        => Io.WriteTo(output, _bitWidth, _dim ?? 0, _nVectors,
            new ReadOnlySpan<byte>(_packedCodes, 0, _packedLength),
            new ReadOnlySpan<float>(_scales, 0, _nVectors), _tqplusShift, _tqplusScale);

    /// <summary>
    /// Durably commits this index to a v8 file. New complete 32-row units are
    /// appended when the existing sync cursor is current; other changes use
    /// an atomic full rewrite. Writers using TurboVector as the same user on
    /// the same host and normalized path coordinate to detect a stale cursor.
    /// </summary>
    public void Sync(string path)
    {
        SyncCore(path, 0, default);
    }

    public static TurboQuantIndex Load(string path)
    {
        LoadedIndex data = Io.LoadTurboQuantIndex(path);
        var index = FromParts(data.Dim == 0 ? null : data.Dim, data.BitWidth, data.VectorCount,
            data.PackedCodes, data.Scales, data.TqPlusShift, data.TqPlusScale);
        index.BindSyncState(data, path);
        return index;
    }

    public static TurboQuantIndex Load(Stream input)
    {
        LoadedIndex data = Io.LoadTurboQuantIndex(input);
        return FromParts(data.Dim == 0 ? null : data.Dim, data.BitWidth, data.VectorCount,
            data.PackedCodes, data.Scales, data.TqPlusShift, data.TqPlusScale);
    }

    public static TurboQuantIndex Load(ReadOnlySpan<byte> bytes)
    {
        LoadedIndex data = Io.LoadTurboQuantIndex(bytes);
        return FromParts(data.Dim == 0 ? null : data.Dim, data.BitWidth, data.VectorCount,
            data.PackedCodes, data.Scales, data.TqPlusShift, data.TqPlusScale);
    }

    internal void WriteIdMapSnapshot(string path, ReadOnlySpan<ulong> ids, Durability durability)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (ids.Length != _nVectors)
        {
            throw new ArgumentException("ID count does not match the index length.", nameof(ids));
        }

        lock (_syncLock)
        {
            Io.WriteIdMapIndex(path, _bitWidth, _dim ?? 0, _nVectors,
                new ReadOnlySpan<byte>(_packedCodes, 0, _packedLength),
                new ReadOnlySpan<float>(_scales, 0, _nVectors), ids,
                _tqplusShift, _tqplusScale, durability);
            ForgetSyncBindingIfPathMatches(path);
        }
    }

    internal static TurboQuantIndex FromParts(
        int? dim, int bitWidth, int nVectors, byte[] packedCodes, float[] scales,
        float[]? tqPlusShift = null, float[]? tqPlusScale = null)
        => new(dim, bitWidth, nVectors, packedCodes, scales, tqPlusShift, tqPlusScale);

    internal void SyncWithIds(string path, ReadOnlySpan<ulong> ids)
    {
        if (ids.Length != _nVectors)
        {
            throw new ArgumentException("ID count does not match the index length.", nameof(ids));
        }

        SyncCore(path, 1, ids);
    }

    private void SyncCore(string path, int kind, ReadOnlySpan<ulong> ids)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        int dim = _dim ?? throw new InvalidOperationException(
            "Cannot sync a lazy index that has never seen an add or calibrate.");
        string fullPath = Path.GetFullPath(path);
        lock (_syncLock)
        {
            bool sameTarget = _syncCursor is not null && PathsEqual(_syncPath, fullPath);
            if (sameTarget)
            {
                SyncCursor cursor = _syncCursor!.Value;
                if (_syncAppendOnly && _syncAppendCompatible
                    && _nVectors >= cursor.VectorCount
                    && (_nVectors / Constants.Block - cursor.VectorCount / Constants.Block) <= 1024)
                {
                    _syncCursor = PersistenceV8.AppendSync(
                        fullPath, kind, _bitWidth, dim, cursor.VectorCount, _nVectors,
                        new ReadOnlySpan<byte>(_packedCodes, 0, _packedLength),
                        new ReadOnlySpan<float>(_scales, 0, _nVectors), ids,
                        _tqplusShift, _tqplusScale, cursor);
                    _syncAppendOnly = true;
                    _syncAppendCompatible = true;
                    return;
                }
            }

            SyncCursor written = PersistenceV8.WriteSyncFile(
                fullPath, kind, _bitWidth, dim, _nVectors,
                new ReadOnlySpan<byte>(_packedCodes, 0, _packedLength),
                new ReadOnlySpan<float>(_scales, 0, _nVectors), ids,
                _tqplusShift, _tqplusScale, sameTarget ? _syncCursor : null);
            _syncPath = fullPath;
            _syncCursor = written;
            _syncAppendOnly = true;
            _syncAppendCompatible = true;
        }
    }

    internal void BindSyncState(LoadedIndex data, string path)
    {
        if (!data.IsV8 || data.Nonce == 0)
        {
            return;
        }

        _syncPath = Path.GetFullPath(path);
        _syncCursor = new SyncCursor(data.Nonce, data.Generation, data.VectorCount);
        _syncAppendOnly = data.AppendCompatible;
        _syncAppendCompatible = data.AppendCompatible;
    }

    private void ForgetSyncBindingIfPathMatches(string path)
    {
        if (_syncCursor is not null && PathsEqual(_syncPath, Path.GetFullPath(path)))
        {
            _syncPath = null;
            _syncCursor = null;
            _syncAppendOnly = false;
            _syncAppendCompatible = false;
        }
    }

    private static bool PathsEqual(string? left, string right)
        => string.Equals(
            left,
            right,
            OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);

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
        _syncAppendOnly = false;
        return last;
    }

    public int Len => _nVectors;

    public bool IsEmpty => _nVectors == 0;

    public int Dim => _dim ?? 0;

    public int? DimOpt => _dim;

    public int BitWidth => _bitWidth;

    public CalibrationState CalibrationState
        => _tqplusShift.Length == 0 ? CalibrationState.Uncalibrated : CalibrationState.Calibrated;

    public ReadOnlySpan<byte> PackedCodes => new(_packedCodes, 0, _packedLength);

    public ReadOnlySpan<float> Scales => new(_scales, 0, _nVectors);

    public ReadOnlySpan<float> TqPlusShift => _tqplusShift;

    public ReadOnlySpan<float> TqPlusScale => _tqplusScale;

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

    private static Lazy<Rotation> MakeRotationLazy(int dim) =>
        new(() => Rotation.Create(dim), LazyThreadSafetyMode.ExecutionAndPublication);

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
        => InputValidation.ValidateBitWidth(bitWidth, nameof(bitWidth));

    private static void ValidateDim(int dim)
        => InputValidation.ValidateDimension(dim, nameof(dim));
}
