using System.Buffers;
using System.Runtime.InteropServices;

namespace TurboVector;

public sealed class IdMapIndex
{
    private readonly TurboQuantIndex _inner;
    private readonly List<ulong> _slotToId;
    private readonly Dictionary<ulong, int> _idToSlot;

    public IdMapIndex(int dim, int bitWidth)
        : this(new TurboQuantIndex(dim, bitWidth), [], [])
    {
    }

    public static IdMapIndex NewLazy(int bitWidth)
        => new(TurboQuantIndex.NewLazy(bitWidth), [], []);

    public void AddWithIds(ReadOnlySpan<float> vectors, ReadOnlySpan<ulong> ids)
    {
        if (_inner.DimOpt is null)
        {
            throw new InvalidOperationException("dim is not set. Use AddWithIds2D on a lazy index to commit the dimension first.");
        }

        AddWithIds2D(vectors, _inner.Dim, ids);
    }

    public void AddWithIds2D(ReadOnlySpan<float> vectors, int dim, ReadOnlySpan<ulong> ids)
    {
        InputValidation.ValidateDimension(dim, nameof(dim));

        if (_inner.DimOpt is int existingDim && existingDim != dim)
        {
            throw new ArgumentException($"Index dimension {existingDim} does not match provided dim {dim}.", nameof(dim));
        }

        if (vectors.Length % dim != 0)
        {
            throw new ArgumentException("vectors length must be divisible by dim.", nameof(vectors));
        }

        int n = vectors.Length / dim;
        if (n != ids.Length)
        {
            throw new ArgumentException($"Expected {n} ids, got {ids.Length}.", nameof(ids));
        }

        if (ids.Length == 1)
        {
            ulong id = ids[0];
            if (_idToSlot.ContainsKey(id))
            {
                throw new InvalidOperationException($"id {id} already present in index");
            }

            int slot = _inner.Len;
            _inner.Add2D(vectors, dim);
            _idToSlot.EnsureCapacity(_idToSlot.Count + 1);
            _slotToId.EnsureCapacity(_slotToId.Count + 1);
            _idToSlot[id] = slot;
            _slotToId.Add(id);
            return;
        }

        var seen = new HashSet<ulong>(ids.Length);
        for (int i = 0; i < ids.Length; i++)
        {
            ulong id = ids[i];
            if (!seen.Add(id))
            {
                throw new InvalidOperationException($"id {id} appears more than once in the batch");
            }

            if (_idToSlot.ContainsKey(id))
            {
                throw new InvalidOperationException($"id {id} already present in index");
            }
        }

        int baseSlot = _inner.Len;
        _inner.Add2D(vectors, dim);

        _idToSlot.EnsureCapacity(_idToSlot.Count + ids.Length);
        _slotToId.EnsureCapacity(_slotToId.Count + ids.Length);
        for (int i = 0; i < ids.Length; i++)
        {
            ulong id = ids[i];
            _idToSlot[id] = baseSlot + i;
            _slotToId.Add(id);
        }
    }

    public bool Remove(ulong id)
    {
        if (!_idToSlot.Remove(id, out int slot))
        {
            return false;
        }

        int last = _slotToId.Count - 1;
        _inner.SwapRemove(slot); // always moves the last slot into position `slot`

        if (slot != last)
        {
            ulong movedId = _slotToId[last];
            _slotToId[slot] = movedId;
            _idToSlot[movedId] = slot;
        }

        _slotToId.RemoveAt(last);
        return true;
    }

    public (float[] Scores, ulong[] Ids) Search(ReadOnlySpan<float> queries, int k)
        => SearchWithAllowlist(queries, k, null);

    public (float[] Scores, ulong[] Ids) SearchWithAllowlist(ReadOnlySpan<float> queries, int k, ulong[]? allowlist)
    {
        int len = _inner.Len;
        ReadOnlySpan<bool> maskSpan = default;
        bool[]? mask = null;
        SearchResults res;
        try
        {
            if (allowlist is not null)
            {
                if (allowlist.Length == 0)
                {
                    throw new ArgumentException("allowlist is empty; pass null to search all vectors instead.", nameof(allowlist));
                }

                mask = ArrayPool<bool>.Shared.Rent(len);
                mask.AsSpan(0, len).Clear();
                for (int i = 0; i < allowlist.Length; i++)
                {
                    ulong id = allowlist[i];
                    if (!_idToSlot.TryGetValue(id, out int slot))
                    {
                        throw new ArgumentException($"id {id} in allowlist is not present in index", nameof(allowlist));
                    }

                    mask[slot] = true;
                }

                maskSpan = mask.AsSpan(0, len);
            }

            res = _inner.SearchWithMask(queries, k, maskSpan);
        }
        finally
        {
            if (mask is not null) ArrayPool<bool>.Shared.Return(mask);
        }

        var slotSpan = CollectionsMarshal.AsSpan(_slotToId);
        ulong[] ids = new ulong[res.Indices.Length];
        for (int i = 0; i < res.Indices.Length; i++)
        {
            long idx = res.Indices[i];
            if (idx < 0)
            {
                // A sentinel -1 here means the search returned fewer valid results than
                // effectiveK, which indicates an internal scoring bug (e.g. BlockHasAllowed
                // incorrectly skipping blocks). Surface it clearly rather than crashing with
                // an IndexOutOfRangeException.
                throw new InvalidOperationException(
                    $"Search returned sentinel index {idx} at result position {i}. " +
                    "This indicates an internal scoring inconsistency.");
            }

            ids[i] = slotSpan[checked((int)idx)];
        }

        return (res.Scores, ids);
    }

    public bool Contains(ulong id) => _idToSlot.ContainsKey(id);

    public void Calibrate(ReadOnlySpan<float> sample) => _inner.Calibrate(sample);

    public void Calibrate2D(ReadOnlySpan<float> sample, int dim) => _inner.Calibrate2D(sample, dim);

    public int Len => _slotToId.Count;

    public bool IsEmpty => _slotToId.Count == 0;

    public int Dim => _inner.Dim;

    public int? DimOpt => _inner.DimOpt;

    public int BitWidth => _inner.BitWidth;

    public CalibrationState CalibrationState => _inner.CalibrationState;

    public ReadOnlySpan<float> TqPlusShift => _inner.TqPlusShift;

    public ReadOnlySpan<float> TqPlusScale => _inner.TqPlusScale;

    public void Prepare() => _inner.Prepare();

    public void Write(string path)
        => Write(path, Durability.Durable);

    public void Write(string path, Durability durability)
        => _inner.WriteIdMapSnapshot(path, CollectionsMarshal.AsSpan(_slotToId), durability);

    /// <summary>Durably commits this index, appending complete units when possible.</summary>
    public void Sync(string path) => _inner.SyncWithIds(path, CollectionsMarshal.AsSpan(_slotToId));

    public byte[] ToBytes()
        => Io.ToIdMapBytes(_inner.BitWidth, _inner.DimOpt ?? 0, _inner.Len,
            _inner.PackedCodes, _inner.Scales, CollectionsMarshal.AsSpan(_slotToId),
            _inner.TqPlusShift, _inner.TqPlusScale);

    public void WriteTo(Stream output)
        => Io.WriteIdMapTo(output, _inner.BitWidth, _inner.DimOpt ?? 0, _inner.Len,
            _inner.PackedCodes, _inner.Scales, CollectionsMarshal.AsSpan(_slotToId),
            _inner.TqPlusShift, _inner.TqPlusScale);

    public static IdMapIndex Load(string path)
    {
        LoadedIndex data = Io.LoadIdMapIndex(path);
        int? dimOpt = data.Dim == 0 ? null : data.Dim;
        var inner = TurboQuantIndex.FromParts(dimOpt, data.BitWidth, data.VectorCount,
            data.PackedCodes, data.Scales, data.TqPlusShift, data.TqPlusScale);
        inner.BindSyncState(data, path);

        var idToSlot = new Dictionary<ulong, int>(data.VectorCount);
        for (int i = 0; i < data.Ids.Length; i++)
        {
            if (!idToSlot.TryAdd(data.Ids[i], i))
            {
                throw new InvalidDataException($"duplicate id {data.Ids[i]} at slot {i} in index file");
            }
        }

        return new IdMapIndex(inner, [.. data.Ids], idToSlot);
    }

    public static IdMapIndex Load(Stream input)
    {
        LoadedIndex data = Io.LoadIdMapIndex(input);
        int? dimOpt = data.Dim == 0 ? null : data.Dim;
        var inner = TurboQuantIndex.FromParts(dimOpt, data.BitWidth, data.VectorCount,
            data.PackedCodes, data.Scales, data.TqPlusShift, data.TqPlusScale);

        var idToSlot = new Dictionary<ulong, int>(data.VectorCount);
        for (int i = 0; i < data.Ids.Length; i++)
        {
            if (!idToSlot.TryAdd(data.Ids[i], i))
            {
                throw new InvalidDataException($"duplicate id {data.Ids[i]} at slot {i} in index image");
            }
        }

        return new IdMapIndex(inner, [.. data.Ids], idToSlot);
    }

    public static IdMapIndex Load(ReadOnlySpan<byte> bytes)
    {
        LoadedIndex data = Io.LoadIdMapIndex(bytes);
        int? dimOpt = data.Dim == 0 ? null : data.Dim;
        var inner = TurboQuantIndex.FromParts(dimOpt, data.BitWidth, data.VectorCount,
            data.PackedCodes, data.Scales, data.TqPlusShift, data.TqPlusScale);

        var idToSlot = new Dictionary<ulong, int>(data.VectorCount);
        for (int i = 0; i < data.Ids.Length; i++)
        {
            if (!idToSlot.TryAdd(data.Ids[i], i))
            {
                throw new InvalidDataException($"duplicate id {data.Ids[i]} at slot {i} in index image");
            }
        }

        return new IdMapIndex(inner, [.. data.Ids], idToSlot);
    }

    private IdMapIndex(TurboQuantIndex inner, List<ulong> slotToId, Dictionary<ulong, int> idToSlot)
    {
        _inner = inner;
        _slotToId = slotToId;
        _idToSlot = idToSlot;
    }
}
