using System.Buffers.Binary;

namespace TurboVector;

public static class Io
{
    private static ReadOnlySpan<byte> TvMagic   => "TVPI"u8;
    private static ReadOnlySpan<byte> TvimMagic => "TVIM"u8;
    private const byte TvVersion = 2;
    private const byte TvimVersion = 2;

    public static void Write(string path, int bitWidth, int dim, int nVectors, ReadOnlySpan<byte> packedCodes, ReadOnlySpan<float> scales)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);

        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, bufferSize: 1 << 16);
        using var writer = new BinaryWriter(stream);
        writer.Write(TvMagic);
        writer.Write(TvVersion);
        WriteCore(writer, bitWidth, dim, nVectors, packedCodes, scales);
    }

    public static (int BitWidth, int Dim, int VectorCount, byte[] PackedCodes, float[] Scales) Load(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);

        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 1 << 16);
        using var reader = new BinaryReader(stream);
        var magic = ReadExactly(reader, 4);
        // Old format (turbovec ≤ 0.4.3) had no magic bytes; the first byte was the
        // bit-width stored as a raw integer (2, 3, or 4). Detect and reject it early.
        if (magic[0] is >= 2 and <= 4)
        {
            throw new InvalidDataException(
                "File format from turbovec ≤ 0.4.3 detected. Rebuild the index with a current version of turbovec to continue.");
        }

        if (!magic.AsSpan().SequenceEqual(TvMagic))
        {
            throw new InvalidDataException($"wrong magic: expected TVPI, got {System.Text.Encoding.ASCII.GetString(magic)}.");
        }

        var version = reader.ReadByte();
        if (version != TvVersion)
        {
            throw new InvalidDataException($"unsupported .tv version {version}; expected {TvVersion}.");
        }

        return ReadCore(reader);
    }

    public static void WriteIdMap(string path, int bitWidth, int dim, int nVectors, ReadOnlySpan<byte> packedCodes, ReadOnlySpan<float> scales, ReadOnlySpan<ulong> slotToId)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);

        if (slotToId.Length != nVectors)
        {
            throw new ArgumentException($"Expected slotToId length {nVectors}, got {slotToId.Length}.", nameof(slotToId));
        }

        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, bufferSize: 1 << 16);
        using var writer = new BinaryWriter(stream);
        writer.Write(TvimMagic);
        writer.Write(TvimVersion);
        WriteCore(writer, bitWidth, dim, nVectors, packedCodes, scales);

        foreach (var id in slotToId)
        {
            WriteUInt64LittleEndian(writer, id);
        }
    }

    public static (int BitWidth, int Dim, int VectorCount, byte[] PackedCodes, float[] Scales, ulong[] SlotToId) LoadIdMap(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);

        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 1 << 16);
        using var reader = new BinaryReader(stream);
        var magic = ReadExactly(reader, 4);
        if (!magic.AsSpan().SequenceEqual(TvimMagic))
        {
            throw new InvalidDataException("invalid .tvim magic; expected TVIM.");
        }

        var version = reader.ReadByte();
        if (version == 1)
        {
            throw new InvalidDataException(
                "File format from turbovec ≤ 0.4.3 detected. Rebuild the index with a current version of turbovec to continue.");
        }

        if (version != TvimVersion)
        {
            throw new InvalidDataException($"unsupported .tvim version {version}; expected {TvimVersion}.");
        }

        var (bitWidth, dim, vectorCount, packedCodes, scales) = ReadCore(reader);
        var slotToId = new ulong[vectorCount];
        for (var i = 0; i < vectorCount; i++)
        {
            slotToId[i] = ReadUInt64LittleEndian(reader);
        }

        return (bitWidth, dim, vectorCount, packedCodes, scales, slotToId);
    }

    private static void WriteCore(BinaryWriter writer, int bitWidth, int dim, int nVectors, ReadOnlySpan<byte> packedCodes, ReadOnlySpan<float> scales)
    {
        ValidateCore(bitWidth, dim, nVectors, packedCodes, scales);

        writer.Write((byte)bitWidth);
        WriteUInt32LittleEndian(writer, checked((uint)dim));
        WriteUInt32LittleEndian(writer, checked((uint)nVectors));
        writer.Write(packedCodes);
        foreach (var scale in scales)
        {
            WriteSingleLittleEndian(writer, scale);
        }
    }

    private static (int BitWidth, int Dim, int VectorCount, byte[] PackedCodes, float[] Scales) ReadCore(BinaryReader reader)
    {
        var bitWidth = reader.ReadByte();
        var dim = checked((int)ReadUInt32LittleEndian(reader));
        var vectorCount = checked((int)ReadUInt32LittleEndian(reader));

        if (dim < 0 || dim % 8 != 0)
        {
            throw new InvalidDataException("dim must be divisible by 8.");
        }

        var packedCodeBytes = checked(dim / 8 * bitWidth * vectorCount);
        var packedCodes = ReadExactly(reader, packedCodeBytes);
        var scales = new float[vectorCount];
        for (var i = 0; i < vectorCount; i++)
        {
            scales[i] = ReadSingleLittleEndian(reader);
        }

        return (bitWidth, dim, vectorCount, packedCodes, scales);
    }

    private static void ValidateCore(int bitWidth, int dim, int nVectors, ReadOnlySpan<byte> packedCodes, ReadOnlySpan<float> scales)
    {
        if (bitWidth <= 0 || bitWidth > byte.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(bitWidth));
        }

        if (dim < 0 || dim % 8 != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dim), "dim must be non-negative and divisible by 8.");
        }

        ArgumentOutOfRangeException.ThrowIfNegative(nVectors);

        var expectedPackedLength = checked(dim / 8 * bitWidth * nVectors);
        if (packedCodes.Length != expectedPackedLength)
        {
            throw new ArgumentException($"Expected packedCodes length {expectedPackedLength}, got {packedCodes.Length}.", nameof(packedCodes));
        }

        if (scales.Length != nVectors)
        {
            throw new ArgumentException($"Expected scales length {nVectors}, got {scales.Length}.", nameof(scales));
        }
    }

    private static byte[] ReadExactly(BinaryReader reader, int byteCount)
    {
        var bytes = reader.ReadBytes(byteCount);
        if (bytes.Length != byteCount)
        {
            throw new EndOfStreamException($"Expected to read {byteCount} bytes, got {bytes.Length}.");
        }

        return bytes;
    }

    private static void WriteUInt32LittleEndian(BinaryWriter writer, uint value)
    {
        Span<byte> buffer = stackalloc byte[sizeof(uint)];
        BinaryPrimitives.WriteUInt32LittleEndian(buffer, value);
        writer.Write(buffer);
    }

    private static uint ReadUInt32LittleEndian(BinaryReader reader)
    {
        Span<byte> buffer = stackalloc byte[sizeof(uint)];
        reader.BaseStream.ReadExactly(buffer);
        return BinaryPrimitives.ReadUInt32LittleEndian(buffer);
    }

    private static void WriteUInt64LittleEndian(BinaryWriter writer, ulong value)
    {
        Span<byte> buffer = stackalloc byte[sizeof(ulong)];
        BinaryPrimitives.WriteUInt64LittleEndian(buffer, value);
        writer.Write(buffer);
    }

    private static ulong ReadUInt64LittleEndian(BinaryReader reader)
    {
        Span<byte> buffer = stackalloc byte[sizeof(ulong)];
        reader.BaseStream.ReadExactly(buffer);
        return BinaryPrimitives.ReadUInt64LittleEndian(buffer);
    }

    private static void WriteSingleLittleEndian(BinaryWriter writer, float value)
    {
        WriteUInt32LittleEndian(writer, unchecked((uint)BitConverter.SingleToInt32Bits(value)));
    }

    private static float ReadSingleLittleEndian(BinaryReader reader)
    {
        return BitConverter.Int32BitsToSingle(unchecked((int)ReadUInt32LittleEndian(reader)));
    }
}
