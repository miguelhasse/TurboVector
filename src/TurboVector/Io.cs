namespace TurboVector;

/// <summary>Controls whether a completed snapshot is flushed to stable storage.</summary>
public enum Durability
{
    /// <summary>Flush file contents before atomically replacing the destination.</summary>
    Durable,
    /// <summary>Atomically replace the destination without forcing a disk flush.</summary>
    Fast,
}

/// <summary>Data stored in a Rust-compatible TurboVector v7/v8 index image.</summary>
public sealed record IndexFileData(
    int BitWidth,
    int Dim,
    int VectorCount,
    byte[] PackedCodes,
    float[] Scales,
    ulong[] Ids,
    float[] TqPlusShift,
    float[] TqPlusScale);

/// <summary>Low-level Rust-compatible index snapshot serialization.</summary>
public static class Io
{
    public static void Write(
        string path,
        int bitWidth,
        int dim,
        int nVectors,
        ReadOnlySpan<byte> packedCodes,
        ReadOnlySpan<float> scales,
        Durability durability = Durability.Durable)
        => PersistenceV8.WriteFile(path, 0, bitWidth, dim, nVectors, packedCodes, scales, default, default, default, durability);

    public static void Write(
        string path,
        int bitWidth,
        int dim,
        int nVectors,
        ReadOnlySpan<byte> packedCodes,
        ReadOnlySpan<float> scales,
        ReadOnlySpan<float> tqPlusShift,
        ReadOnlySpan<float> tqPlusScale,
        Durability durability = Durability.Durable)
        => PersistenceV8.WriteFile(path, 0, bitWidth, dim, nVectors, packedCodes, scales, default,
            tqPlusShift, tqPlusScale, durability);

    public static (int BitWidth, int Dim, int VectorCount, byte[] PackedCodes, float[] Scales) Load(string path)
    {
        LoadedIndex data = PersistenceV8.LoadFile(path, 0);
        if (data.TqPlusShift.Length != 0)
        {
            throw new InvalidDataException("This low-level tuple omits TQ+ calibration; use Io.LoadData to preserve it.");
        }

        return (data.BitWidth, data.Dim, data.VectorCount, data.PackedCodes, data.Scales);
    }

    public static IndexFileData LoadData(string path)
        => ToPublicData(PersistenceV8.LoadFile(path, 0));

    public static void WriteIdMap(
        string path,
        int bitWidth,
        int dim,
        int nVectors,
        ReadOnlySpan<byte> packedCodes,
        ReadOnlySpan<float> scales,
        ReadOnlySpan<ulong> slotToId,
        Durability durability = Durability.Durable)
        => PersistenceV8.WriteFile(path, 1, bitWidth, dim, nVectors, packedCodes, scales, slotToId,
            default, default, durability);

    public static void WriteIdMap(
        string path,
        int bitWidth,
        int dim,
        int nVectors,
        ReadOnlySpan<byte> packedCodes,
        ReadOnlySpan<float> scales,
        ReadOnlySpan<ulong> slotToId,
        ReadOnlySpan<float> tqPlusShift,
        ReadOnlySpan<float> tqPlusScale,
        Durability durability = Durability.Durable)
        => PersistenceV8.WriteFile(path, 1, bitWidth, dim, nVectors, packedCodes, scales, slotToId,
            tqPlusShift, tqPlusScale, durability);

    public static (int BitWidth, int Dim, int VectorCount, byte[] PackedCodes, float[] Scales, ulong[] SlotToId) LoadIdMap(string path)
    {
        LoadedIndex data = PersistenceV8.LoadFile(path, 1);
        if (data.TqPlusShift.Length != 0)
        {
            throw new InvalidDataException("This low-level tuple omits TQ+ calibration; use Io.LoadIdMapData to preserve it.");
        }

        return (data.BitWidth, data.Dim, data.VectorCount, data.PackedCodes, data.Scales, data.Ids);
    }

    public static IndexFileData LoadIdMapData(string path)
        => ToPublicData(PersistenceV8.LoadFile(path, 1));

    public static byte[] ToBytes(
        int bitWidth,
        int dim,
        int nVectors,
        ReadOnlySpan<byte> packedCodes,
        ReadOnlySpan<float> scales,
        ReadOnlySpan<float> tqPlusShift = default,
        ReadOnlySpan<float> tqPlusScale = default)
        => PersistenceV8.ToBytes(0, bitWidth, dim, nVectors, packedCodes, scales, default, tqPlusShift, tqPlusScale);

    public static byte[] ToIdMapBytes(
        int bitWidth,
        int dim,
        int nVectors,
        ReadOnlySpan<byte> packedCodes,
        ReadOnlySpan<float> scales,
        ReadOnlySpan<ulong> slotToId,
        ReadOnlySpan<float> tqPlusShift = default,
        ReadOnlySpan<float> tqPlusScale = default)
        => PersistenceV8.ToBytes(1, bitWidth, dim, nVectors, packedCodes, scales, slotToId, tqPlusShift, tqPlusScale);

    public static void WriteTo(
        Stream output,
        int bitWidth,
        int dim,
        int nVectors,
        ReadOnlySpan<byte> packedCodes,
        ReadOnlySpan<float> scales,
        ReadOnlySpan<float> tqPlusShift = default,
        ReadOnlySpan<float> tqPlusScale = default)
        => PersistenceV8.WriteTo(output, 0, bitWidth, dim, nVectors, packedCodes, scales, default, tqPlusShift, tqPlusScale);

    public static void WriteIdMapTo(
        Stream output,
        int bitWidth,
        int dim,
        int nVectors,
        ReadOnlySpan<byte> packedCodes,
        ReadOnlySpan<float> scales,
        ReadOnlySpan<ulong> slotToId,
        ReadOnlySpan<float> tqPlusShift = default,
        ReadOnlySpan<float> tqPlusScale = default)
        => PersistenceV8.WriteTo(output, 1, bitWidth, dim, nVectors, packedCodes, scales, slotToId, tqPlusShift, tqPlusScale);

    public static IndexFileData LoadData(Stream input)
        => ToPublicData(PersistenceV8.Load(input, 0));

    public static IndexFileData LoadIdMapData(Stream input)
        => ToPublicData(PersistenceV8.Load(input, 1));

    public static IndexFileData LoadData(ReadOnlySpan<byte> bytes)
        => ToPublicData(PersistenceV8.Load(bytes, 0));

    public static IndexFileData LoadIdMapData(ReadOnlySpan<byte> bytes)
        => ToPublicData(PersistenceV8.Load(bytes, 1));

    internal static LoadedIndex LoadTurboQuantIndex(string path) => PersistenceV8.LoadFile(path, 0);

    internal static LoadedIndex LoadIdMapIndex(string path) => PersistenceV8.LoadFile(path, 1);

    internal static LoadedIndex LoadTurboQuantIndex(Stream stream) => PersistenceV8.Load(stream, 0);

    internal static LoadedIndex LoadTurboQuantIndex(ReadOnlySpan<byte> bytes) => PersistenceV8.Load(bytes, 0);

    internal static LoadedIndex LoadIdMapIndex(Stream stream) => PersistenceV8.Load(stream, 1);

    internal static LoadedIndex LoadIdMapIndex(ReadOnlySpan<byte> bytes) => PersistenceV8.Load(bytes, 1);

    internal static void WriteTurboQuantIndex(
        string path, int bitWidth, int dim, int nVectors,
        ReadOnlySpan<byte> packedCodes, ReadOnlySpan<float> scales,
        ReadOnlySpan<float> shift, ReadOnlySpan<float> tqScale, Durability durability)
        => PersistenceV8.WriteFile(path, 0, bitWidth, dim, nVectors, packedCodes, scales, default, shift, tqScale, durability);

    internal static void WriteIdMapIndex(
        string path, int bitWidth, int dim, int nVectors,
        ReadOnlySpan<byte> packedCodes, ReadOnlySpan<float> scales, ReadOnlySpan<ulong> ids,
        ReadOnlySpan<float> shift, ReadOnlySpan<float> tqScale, Durability durability)
        => PersistenceV8.WriteFile(path, 1, bitWidth, dim, nVectors, packedCodes, scales, ids, shift, tqScale, durability);

    private static IndexFileData ToPublicData(LoadedIndex data)
        => new(data.BitWidth, data.Dim, data.VectorCount, data.PackedCodes, data.Scales,
            data.Ids, data.TqPlusShift, data.TqPlusScale);
}
