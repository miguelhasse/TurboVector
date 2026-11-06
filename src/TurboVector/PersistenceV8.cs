using System.Buffers.Binary;
using System.Security.Cryptography;

namespace TurboVector;

internal sealed record LoadedIndex(
    int BitWidth,
    int Dim,
    int VectorCount,
    byte[] PackedCodes,
    float[] Scales,
    ulong[] Ids,
    float[] TqPlusShift,
    float[] TqPlusScale,
    ulong Nonce,
    ulong Generation,
    bool IsV8,
    bool AppendCompatible);

internal readonly record struct SyncCursor(ulong Nonce, ulong Generation, int VectorCount);

internal static class PersistenceV8
{
    private static ReadOnlySpan<byte> V7Magic => "TV7\0"u8;
    private static ReadOnlySpan<byte> V8Magic => "TV8\0"u8;
    private const byte V7Version = 2;
    private const byte V8Version = 1;
    private const int MaxOps = 1024;
    private const int Block = 32;
    private const byte SequentialLayout = 0;
    private const byte PlanesLayout = 1;
    private const float MaxVectorScale = 1e22f;

    private static SyncTargetLock LockSyncTarget(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        string fullPath = Path.GetFullPath(path);
        if (OperatingSystem.IsWindows())
        {
            fullPath = fullPath.ToUpperInvariant();
        }

        byte[] hash = SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(fullPath));
        string name = "TurboVector-" + Convert.ToHexString(hash);
        var options = new NamedWaitHandleOptions
        {
            CurrentSessionOnly = false,
            CurrentUserOnly = true
        };
        var mutex = new Mutex(initiallyOwned: false, name, options);
        try
        {
            try
            {
                mutex.WaitOne();
            }
            catch (AbandonedMutexException error)
            {
                mutex.ReleaseMutex();
                throw new IOException($"The writer lock for '{path}' was abandoned. Retry after inspecting the file.", error);
            }

            return new SyncTargetLock(mutex);
        }
        catch
        {
            mutex.Dispose();
            throw;
        }
    }

    private sealed class SyncTargetLock(Mutex mutex) : IDisposable
    {
        public void Dispose()
        {
            try
            {
                mutex.ReleaseMutex();
            }
            finally
            {
                mutex.Dispose();
            }
        }
    }

    internal static byte[] ToBytes(
        int kind,
        int bitWidth,
        int dim,
        int nVectors,
        ReadOnlySpan<byte> packedCodes,
        ReadOnlySpan<float> scales,
        ReadOnlySpan<ulong> ids,
        ReadOnlySpan<float> tqPlusShift,
        ReadOnlySpan<float> tqPlusScale)
    {
        Validate(kind, bitWidth, dim, nVectors, packedCodes, scales, ids, tqPlusShift, tqPlusScale);
        byte[] image = new byte[GetImageLength(dim, bitWidth, nVectors, kind, tqPlusShift.Length)];
        using var output = new MemoryStream(image, writable: true);
        WriteImage(output, kind, bitWidth, dim, nVectors, packedCodes, scales, ids, tqPlusShift, tqPlusScale);
        if (output.Position != image.Length)
        {
            throw new InvalidOperationException("Serialized image length did not match its calculated geometry.");
        }

        return image;
    }

    internal static void WriteTo(
        Stream output,
        int kind,
        int bitWidth,
        int dim,
        int nVectors,
        ReadOnlySpan<byte> packedCodes,
        ReadOnlySpan<float> scales,
        ReadOnlySpan<ulong> ids,
        ReadOnlySpan<float> tqPlusShift,
        ReadOnlySpan<float> tqPlusScale)
    {
        ArgumentNullException.ThrowIfNull(output);
        if (!output.CanWrite)
        {
            throw new ArgumentException("Stream must be writable.", nameof(output));
        }

        Validate(kind, bitWidth, dim, nVectors, packedCodes, scales, ids, tqPlusShift, tqPlusScale);
        WriteImage(output, kind, bitWidth, dim, nVectors, packedCodes, scales, ids, tqPlusShift, tqPlusScale);
    }

    internal static void WriteFile(
        string path,
        int kind,
        int bitWidth,
        int dim,
        int nVectors,
        ReadOnlySpan<byte> packedCodes,
        ReadOnlySpan<float> scales,
        ReadOnlySpan<ulong> ids,
        ReadOnlySpan<float> tqPlusShift,
        ReadOnlySpan<float> tqPlusScale,
        Durability durability,
        ulong nonce = 0)
    {
        using SyncTargetLock targetLock = LockSyncTarget(path);
        WriteFileCore(path, kind, bitWidth, dim, nVectors, packedCodes, scales, ids,
            tqPlusShift, tqPlusScale, durability, nonce);
    }

    private static void WriteFileCore(
        string path,
        int kind,
        int bitWidth,
        int dim,
        int nVectors,
        ReadOnlySpan<byte> packedCodes,
        ReadOnlySpan<float> scales,
        ReadOnlySpan<ulong> ids,
        ReadOnlySpan<float> tqPlusShift,
        ReadOnlySpan<float> tqPlusScale,
        Durability durability,
        ulong nonce)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        Validate(kind, bitWidth, dim, nVectors, packedCodes, scales, ids, tqPlusShift, tqPlusScale);
        if (durability is not (Durability.Durable or Durability.Fast))
        {
            throw new ArgumentOutOfRangeException(nameof(durability));
        }

        string fullPath = Path.GetFullPath(path);
        string directory = Path.GetDirectoryName(fullPath)!;
        string tempPath = Path.Combine(directory, $"{Path.GetFileName(fullPath)}.tmp.{Guid.NewGuid():N}");
        try
        {
            using (var stream = new FileStream(
                tempPath,
                FileMode.CreateNew,
                FileAccess.Write,
                FileShare.None,
                bufferSize: 1 << 20,
                durability == Durability.Durable ? FileOptions.WriteThrough : FileOptions.SequentialScan))
            {
                WriteImage(stream, kind, bitWidth, dim, nVectors, packedCodes, scales, ids, tqPlusShift, tqPlusScale, nonce);
                if (durability == Durability.Durable)
                {
                    stream.Flush(flushToDisk: true);
                }
                else
                {
                    stream.Flush();
                }
            }

            File.Move(tempPath, fullPath, overwrite: true);
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
            }
        }
    }

    internal static SyncCursor WriteSyncFile(
        string path,
        int kind,
        int bitWidth,
        int dim,
        int nVectors,
        ReadOnlySpan<byte> packedCodes,
        ReadOnlySpan<float> scales,
        ReadOnlySpan<ulong> ids,
        ReadOnlySpan<float> tqPlusShift,
        ReadOnlySpan<float> tqPlusScale,
        SyncCursor? expectedCursor)
    {
        using SyncTargetLock targetLock = LockSyncTarget(path);
        if (expectedCursor is SyncCursor cursor)
        {
            VerifySyncCursor(path, kind, bitWidth, dim, cursor);
        }

        Span<byte> nonceBytes = stackalloc byte[sizeof(ulong)];
        ulong nonce;
        do
        {
            RandomNumberGenerator.Fill(nonceBytes);
            nonce = BinaryPrimitives.ReadUInt64LittleEndian(nonceBytes);
        }
        while (nonce == 0);

        WriteFileCore(path, kind, bitWidth, dim, nVectors, packedCodes, scales, ids,
            tqPlusShift, tqPlusScale, Durability.Durable, nonce);
        return new SyncCursor(nonce, 0, nVectors);
    }

    private static void VerifySyncCursor(
        string path,
        int kind,
        int bitWidth,
        int dim,
        SyncCursor cursor)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.None);
        ParsedSyncHeader current = ReadSyncHeader(stream, kind, bitWidth, dim, cursor.Nonce);
        if (current.Generation != cursor.Generation || current.Count != cursor.VectorCount)
        {
            throw ForeignSyncTarget(path);
        }
    }

    internal static SyncCursor AppendSync(
        string path,
        int kind,
        int bitWidth,
        int dim,
        int oldCount,
        int newCount,
        ReadOnlySpan<byte> packedCodes,
        ReadOnlySpan<float> scales,
        ReadOnlySpan<ulong> ids,
        ReadOnlySpan<float> tqPlusShift,
        ReadOnlySpan<float> tqPlusScale,
        SyncCursor cursor)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        using SyncTargetLock targetLock = LockSyncTarget(path);
        if (newCount < oldCount)
        {
            throw new InvalidOperationException("Incremental sync only supports append-only changes.");
        }

        Validate(kind, bitWidth, dim, newCount, packedCodes, scales, ids, tqPlusShift, tqPlusScale);
        if (newCount == oldCount)
        {
            using var unchanged = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.None);
            ParsedSyncHeader current = ReadSyncHeader(unchanged, kind, bitWidth, dim, cursor.Nonce);
            if (current.Generation != cursor.Generation || current.Count != cursor.VectorCount)
            {
                throw ForeignSyncTarget(path);
            }

            return cursor;
        }

        int rowBytes = checked(dim / (8 / bitWidth));
        int idBytes = kind == 1 ? sizeof(ulong) : 0;
        int tailRowBytes = checked(rowBytes + sizeof(float) + idBytes);
        int unitBytes = checked(Block * tailRowBytes);
        int headerBytes = HeaderLength(rowBytes, idBytes);
        int oldBlocks = oldCount / Block;
        int newBlocks = newCount / Block;
        int appendedBlocks = newBlocks - oldBlocks;
        if (appendedBlocks > MaxOps)
        {
            throw new InvalidOperationException("Append batch exceeds the v8 incremental-sync limit; perform a full Write instead.");
        }

        int levels = 1 << bitWidth;
        int calibrationLength;
        int superblockOffset;
        using var stream = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None, 1 << 20);
        ParsedSyncHeader active = ReadSyncHeader(stream, kind, bitWidth, dim, cursor.Nonce);
        if (active.Generation != cursor.Generation || active.Count != cursor.VectorCount)
        {
            throw ForeignSyncTarget(path);
        }

        if (!active.AppendCompatible)
        {
            throw new InvalidOperationException(
                "This v8 commit contains redo or materialized operations; append sync requires a full Write to compact it first.");
        }

        if (cursor.Generation == ulong.MaxValue)
        {
            throw new InvalidOperationException("The v8 sync generation is exhausted.");
        }

        stream.Position = checked(24 + (2 * levels - 1) * sizeof(float));
        Span<byte> uintBytes = stackalloc byte[sizeof(uint)];
        stream.ReadExactly(uintBytes);
        calibrationLength = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(uintBytes));
        superblockOffset = checked(24 + (2 * levels - 1) * sizeof(float) + sizeof(uint)
            + 2 * calibrationLength * sizeof(float) + sizeof(uint));
        int unitOffset = checked(superblockOffset + 2 * headerBytes);
        long committedLength = checked((long)unitOffset + (long)oldBlocks * unitBytes);
        if (stream.Length < committedLength)
        {
            throw Invalid("sync target is shorter than its committed units");
        }

        // Bytes beyond the committed block range may be an interrupted prior append.
        stream.SetLength(committedLength);

        ulong generation = cursor.Generation + 1;
        int firstPackedRowBytes = checked(dim / 8 * bitWidth);
        int tailCount = newCount % Block;
        int tailStart = newCount - tailCount;
        int headerOffset = checked(superblockOffset + (int)(generation & 1) * headerBytes);
        int totalDeltaLength = checked(sizeof(ulong) + appendedBlocks * (sizeof(uint) + unitBytes));
        var digest = new CrcAccumulator(totalDeltaLength);
        Span<byte> generationBytes = stackalloc byte[sizeof(ulong)];
        Span<byte> blockBytes = stackalloc byte[sizeof(uint)];
        BinaryPrimitives.WriteUInt64LittleEndian(generationBytes, generation);
        digest.Push(generationBytes);
        byte[] unit = appendedBlocks == 0 ? [] : new byte[unitBytes];
        byte[] row = tailCount == 0 ? [] : new byte[rowBytes];
        for (int block = oldBlocks; block < newBlocks; block++)
        {
            Array.Clear(unit);
            int codeLength = Block * rowBytes;
            for (int lane = 0; lane < Block; lane++)
            {
                int vector = block * Block + lane;
                ReadOnlySpan<byte> packedRow = packedCodes.Slice(vector * firstPackedRowBytes, firstPackedRowBytes);
                EncodePackedRow(packedRow, bitWidth, dim, unit.AsSpan(0, codeLength), block, lane, rowBytes);
                WriteSingle(unit.AsSpan(codeLength + lane * sizeof(float), sizeof(float)), scales[vector]);
                if (kind == 1)
                {
                    WriteUInt64(unit.AsSpan(codeLength + Block * sizeof(float) + lane * sizeof(ulong), sizeof(ulong)),
                        ids[vector]);
                }
            }

            BinaryPrimitives.WriteUInt32LittleEndian(blockBytes, (uint)block);
            digest.Push(blockBytes);
            digest.Push(unit);
            stream.Position = checked((long)unitOffset + (long)block * unitBytes);
            stream.Write(unit);
        }

        if (appendedBlocks > 0)
        {
            stream.Flush(flushToDisk: true);
        }

        int usedHeaderBytes = checked(16 + tailCount * tailRowBytes + 4 + 4 + 12 + 4);
        var header = new MemoryStream(usedHeaderBytes);
        using (var writer = new BinaryWriter(header, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            WriteUInt64(writer, generation);
            WriteUInt64(writer, checked((ulong)newCount));
            for (int lane = 0; lane < tailCount; lane++)
            {
                int vector = tailStart + lane;
                EncodePackedRow(
                    packedCodes.Slice(vector * firstPackedRowBytes, firstPackedRowBytes),
                    bitWidth,
                    dim,
                    row);
                writer.Write(row);
                WriteSingle(writer, scales[vector]);
                if (kind == 1)
                {
                    WriteUInt64(writer, ids[vector]);
                }
            }

            WriteUInt32(writer, 0); // redo groups
            WriteUInt32(writer, 0); // materialized units
            WriteUInt32(writer, (uint)oldBlocks);
            WriteUInt32(writer, (uint)newBlocks);
            WriteUInt32(writer, digest.Finish());
            byte[] body = header.ToArray();
            WriteUInt32(writer, Crc32(body));
        }

        if (header.Length > headerBytes)
        {
            throw new InvalidOperationException("Incremental commit header exceeds its reserved slot.");
        }

        stream.Position = headerOffset;
        header.Position = 0;
        header.CopyTo(stream);
        stream.Flush(flushToDisk: true);
        return new SyncCursor(cursor.Nonce, generation, newCount);
    }

    private static ParsedSyncHeader ReadSyncHeader(
        Stream stream,
        int kind,
        int bitWidth,
        int dim,
        ulong expectedNonce)
    {
        Span<byte> prefix = stackalloc byte[24];
        stream.Position = 0;
        stream.ReadExactly(prefix);
        if (!prefix[..4].SequenceEqual(V8Magic)
            || prefix[4] != V8Version
            || prefix[5] != bitWidth
            || prefix[6] != kind
            || prefix[7] != SequentialLayout
            || ReadUInt32(prefix, 8) != dim
            || ReadUInt64(prefix, 12) != expectedNonce
            || ReadUInt32(prefix, 20) != MaxOps)
        {
            throw ForeignSyncTarget(null);
        }

        int levels = 1 << bitWidth;
        int rowBytes = checked(dim / (8 / bitWidth));
        int idBytes = kind == 1 ? sizeof(ulong) : 0;
        int tailRowBytes = checked(rowBytes + sizeof(float) + idBytes);
        int unitBytes = checked(Block * tailRowBytes);
        int headerCapacity = HeaderLength(rowBytes, idBytes);
        long calibrationLengthOffset = checked(24L + (2 * levels - 1) * sizeof(float));
        Span<byte> uintBytes = stackalloc byte[sizeof(uint)];
        stream.Position = calibrationLengthOffset;
        stream.ReadExactly(uintBytes);
        int calibrationLength = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(uintBytes));
        if (calibrationLength != 0 && calibrationLength != dim)
        {
            throw ForeignSyncTarget(null);
        }

        int unitOffset = checked(24 + (2 * levels - 1) * sizeof(float) + sizeof(uint)
            + 2 * calibrationLength * sizeof(float) + sizeof(uint) + 2 * headerCapacity);
        int superblockLength = unitOffset - 2 * headerCapacity;
        byte[] superblock = new byte[superblockLength];
        stream.Position = 0;
        stream.ReadExactly(superblock);
        if (Crc32(superblock.AsSpan(0, superblockLength - sizeof(uint)))
            != ReadUInt32(superblock, superblockLength - sizeof(uint)))
        {
            throw Invalid("sync target has a corrupt superblock");
        }

        byte[] header = new byte[headerCapacity];
        var candidates = new List<ParsedSyncHeader>(2);
        for (int slot = 0; slot < 2; slot++)
        {
            long offset = checked((long)(unitOffset - 2 * headerCapacity) + (long)slot * headerCapacity);
            if (stream.Length < offset + headerCapacity)
            {
                continue;
            }

            stream.Position = offset;
            stream.ReadExactly(header);
            ParsedSyncHeader? candidate = ParseSyncHeader(
                stream, header, slot, unitOffset, unitBytes, rowBytes, tailRowBytes, idBytes);
            if (candidate is not null)
            {
                candidates.Add(candidate);
            }
        }

        candidates.Sort(static (left, right) => right.Generation.CompareTo(left.Generation));
        if (candidates.Count == 0)
        {
            throw Invalid("no valid v8 sync header");
        }

        return candidates[0];
    }

    private static ParsedSyncHeader? ParseSyncHeader(
        Stream stream,
        ReadOnlySpan<byte> header,
        int slot,
        int unitOffset,
        int unitBytes,
        int rowBytes,
        int tailRowBytes,
        int idBytes)
    {
        if (header.Length < 16)
        {
            return null;
        }

        ulong generation = ReadUInt64(header, 0);
        if ((generation & 1) != (uint)slot)
        {
            return null;
        }

        ulong count64 = ReadUInt64(header, 8);
        if (count64 > int.MaxValue)
        {
            return null;
        }

        int count = (int)count64;
        int blocks = count / Block;
        long endUnits = (long)unitOffset + (long)blocks * unitBytes;
        if (endUnits > stream.Length)
        {
            return null;
        }

        int position = checked(16 + (count % Block) * tailRowBytes);
        if (position > header.Length - sizeof(uint))
        {
            return null;
        }

        uint groupCount = ReadUInt32(header, position);
        if (groupCount > MaxOps)
        {
            return null;
        }

        position += sizeof(uint);
        int opSize = checked(1 + rowBytes + sizeof(float) + idBytes);
        for (int group = 0; group < groupCount; group++)
        {
            if (position > header.Length - 5)
            {
                return null;
            }

            byte opCount = header[position + 4];
            position = checked(position + 5 + opCount * opSize);
            if (position > header.Length)
            {
                return null;
            }
        }

        if (position > header.Length - sizeof(uint))
        {
            return null;
        }

        uint materializedCount = ReadUInt32(header, position);
        if (materializedCount > MaxOps)
        {
            return null;
        }

        position = checked(position + sizeof(uint) + (int)materializedCount * sizeof(uint));
        if (position > header.Length - 16)
        {
            return null;
        }

        uint appFrom = ReadUInt32(header, position);
        uint appTo = ReadUInt32(header, position + 4);
        uint deltaCrc = ReadUInt32(header, position + 8);
        uint storedCrc = ReadUInt32(header, position + 12);
        if (Crc32(header[..(position + 12)]) != storedCrc || appFrom > appTo || appTo > blocks)
        {
            return null;
        }

        bool appendCompatible = groupCount == 0 && materializedCount == 0;
        if (appendCompatible && !VerifyAppendDelta(
            stream, generation, appFrom, appTo, deltaCrc, unitOffset, unitBytes))
        {
            return null;
        }

        return new ParsedSyncHeader(generation, count, appendCompatible);
    }

    private static bool VerifyAppendDelta(
        Stream stream,
        ulong generation,
        uint appFrom,
        uint appTo,
        uint expectedCrc,
        int unitOffset,
        int unitBytes)
    {
        int blocks = checked((int)(appTo - appFrom));
        int total = checked(sizeof(ulong) + blocks * (sizeof(uint) + unitBytes));
        var digest = new CrcAccumulator(total);
        Span<byte> number = stackalloc byte[sizeof(ulong)];
        Span<byte> blockBytes = stackalloc byte[sizeof(uint)];
        BinaryPrimitives.WriteUInt64LittleEndian(number, generation);
        digest.Push(number);
        byte[] unit = new byte[unitBytes];
        for (uint block = appFrom; block < appTo; block++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(blockBytes, block);
            digest.Push(blockBytes);
            long offset = checked((long)unitOffset + (long)block * unitBytes);
            if (offset > stream.Length - unitBytes)
            {
                return false;
            }

            stream.Position = offset;
            stream.ReadExactly(unit);
            digest.Push(unit);
        }

        return digest.Finish() == expectedCrc;
    }

    private static InvalidDataException ForeignSyncTarget(string? path)
        => new($"The v8 sync target {path ?? "(stream)"} no longer matches the index's committed cursor.");

    private static void WriteZeros(Stream stream, int count)
    {
        byte[] zeros = new byte[Math.Min(count, 64 * 1024)];
        while (count > 0)
        {
            int take = Math.Min(count, zeros.Length);
            stream.Write(zeros, 0, take);
            count -= take;
        }
    }

    internal static LoadedIndex LoadFile(string path, int expectedKind)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16);
        return Load(stream, expectedKind);
    }

    internal static LoadedIndex Load(Stream input, int expectedKind)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (!input.CanRead)
        {
            throw new ArgumentException("Stream must be readable.", nameof(input));
        }

        using var buffer = new MemoryStream();
        input.CopyTo(buffer);
        return Load(buffer.GetBuffer().AsSpan(0, checked((int)buffer.Length)), expectedKind);
    }

    internal static LoadedIndex Load(ReadOnlySpan<byte> raw, int expectedKind)
    {
        try
        {
            return LoadCore(raw, expectedKind);
        }
        catch (OverflowException ex)
        {
            throw new InvalidDataException("Index file contains an out-of-range length or offset.", ex);
        }
        catch (ArgumentOutOfRangeException ex)
        {
            throw new InvalidDataException("Index file contains an invalid length or offset.", ex);
        }
    }

    private static LoadedIndex LoadCore(ReadOnlySpan<byte> raw, int expectedKind)
    {
        if (raw.Length < 12)
        {
            throw Invalid("not a v7 or v8 file");
        }

        bool isV8 = raw[..4].SequenceEqual(V8Magic);
        bool isV7 = raw[..4].SequenceEqual(V7Magic);
        if (!isV7 && !isV8)
        {
            if (raw[..4].SequenceEqual("TVPI"u8) || raw[..4].SequenceEqual("TVIM"u8))
            {
                if (raw[4] == 1)
                {
                    throw Invalid("File format from turbovec ≤ 0.4.3 detected. Rebuild the index with a current version of turbovec to continue.");
                }

                if (raw[4] == 2)
                {
                    throw Invalid($"unsupported legacy index format version {raw[4]}; this build reads Rust v7 and v8 files");
                }
            }

            if (raw[0] is >= 2 and <= 4)
            {
                throw Invalid("File format from turbovec ≤ 0.4.3 detected. Rebuild the index with a current version of turbovec to continue.");
            }

            throw Invalid("wrong magic: expected TV7\\0 or TV8\\0");
        }

        byte version = raw[4];
        if (isV7 && version != V7Version)
        {
            throw Invalid($"unsupported v7 revision {version}");
        }

        if (isV8 && version != V8Version)
        {
            throw Invalid($"unsupported v8 revision {version}");
        }

        int bitWidth = raw[5];
        if (bitWidth is not (2 or 3 or 4))
        {
            throw Invalid($"bit_width {bitWidth} out of range");
        }

        int kind = raw[6];
        if (kind != expectedKind)
        {
            throw Invalid(kind switch
            {
                0 => "this file holds a TurboQuantIndex; load it with TurboQuantIndex.Load",
                1 => "this file holds an IdMapIndex; load it with IdMapIndex.Load",
                _ => $"unknown index kind {kind}",
            });
        }

        byte layout = isV8 ? raw[7] : SequentialLayout;
        if (layout is not (SequentialLayout or PlanesLayout) || (layout == PlanesLayout && bitWidth == 3))
        {
            throw Invalid($"unsupported v{(isV8 ? 8 : 7)} unit layout {layout}");
        }

        int fixedSize = isV8 ? 24 : 23;
        int dim = checked((int)ReadUInt32(raw, fixedSize - 16));
        if (dim != 0 && (dim % 8 != 0 || dim > Constants.MaxDim))
        {
            throw Invalid($"dim {dim} invalid");
        }

        ulong nonce = ReadUInt64(raw, fixedSize - 12);
        if (ReadUInt32(raw, fixedSize - 4) != MaxOps)
        {
            throw Invalid("unsupported redo-op capacity");
        }

        int geometryRowBytes = dim == 0 ? 0 : checked(dim / (8 / bitWidth));
        if (raw.Length < (long)MaxOps * dim * bitWidth / 8)
        {
            throw Invalid("truncated file: the declared geometry cannot fit its reserved header");
        }

        int levels = 1 << bitWidth;
        int offset = fixedSize;
        int codebookBytes = checked((2 * levels - 1) * sizeof(float));
        if (offset > raw.Length - codebookBytes)
        {
            throw Invalid("truncated codebook");
        }

        float[] boundaries;
        float[] centroids;
        if (dim == 0)
        {
            boundaries = new float[levels - 1];
            centroids = new float[levels];
            offset += codebookBytes;
        }
        else
        {
            (boundaries, centroids) = Codebook.Compute(bitWidth, dim);
            foreach (float expected in boundaries)
            {
                if (ReadSingle(raw, offset) != expected)
                {
                    throw Invalid("embedded codebook drifted from the canonical one");
                }

                offset += sizeof(float);
            }

            foreach (float expected in centroids)
            {
                if (ReadSingle(raw, offset) != expected)
                {
                    throw Invalid("embedded codebook drifted from the canonical one");
                }

                offset += sizeof(float);
            }
        }

        int calibrationLength = checked((int)ReadUInt32(raw, offset));
        offset += sizeof(uint);
        if (calibrationLength != 0 && calibrationLength != dim)
        {
            throw Invalid($"calibration length {calibrationLength} != dim {dim}");
        }

        int shiftOffset = offset;
        offset = checked(offset + calibrationLength * sizeof(float));
        int scaleOffset = offset;
        offset = checked(offset + calibrationLength * sizeof(float));
        if (offset > raw.Length - sizeof(uint))
        {
            throw Invalid("truncated superblock");
        }

        float[] shift = new float[calibrationLength];
        float[] tqScale = new float[calibrationLength];
        for (int i = 0; i < calibrationLength; i++)
        {
            shift[i] = ReadSingle(raw, shiftOffset + i * sizeof(float));
            tqScale[i] = ReadSingle(raw, scaleOffset + i * sizeof(float));
        }

        ValidateCalibration(shift, tqScale);
        uint superblockCrc = ReadUInt32(raw, offset);
        if (Crc32(raw[..offset]) != superblockCrc)
        {
            throw Invalid("corrupt superblock (crc mismatch)");
        }

        offset += sizeof(uint);
        int rowBytes = geometryRowBytes;
        if (layout == PlanesLayout && dim != 0 && !PlanesGeometryValid(bitWidth, rowBytes))
        {
            throw Invalid($"dim {dim} has no planes layout at {bitWidth} bits");
        }

        int idBytes = kind == 1 ? sizeof(ulong) : 0;
        int tailRowBytes = checked(rowBytes + sizeof(float) + idBytes);
        int unitBytes = checked(Block * (rowBytes + sizeof(float) + idBytes));
        int headerBytes = HeaderLength(rowBytes, idBytes);
        int unitOffset = checked(offset + 2 * headerBytes);
        int fileLength = raw.Length;
        var candidates = new List<ParsedHeader>(2);
        for (int slot = 0; slot < 2; slot++)
        {
            ParsedHeader? candidate = ParseHeader(raw, offset + slot * headerBytes, slot, fileLength,
                dim, rowBytes, tailRowBytes, idBytes, unitOffset, unitBytes, headerBytes);
            if (candidate is not null)
            {
                candidates.Add(candidate);
            }
        }

        candidates.Sort(static (left, right) => right.Generation.CompareTo(left.Generation));
        ParsedHeader? selected = null;
        foreach (ParsedHeader candidate in candidates)
        {
            if (VerifyDelta(raw, candidate, unitOffset, unitBytes))
            {
                selected = candidate;
                break;
            }
        }

        if (selected is null)
        {
            throw Invalid("no valid commit header — unrecoverable v7/v8 file");
        }

        int count = selected.Count;
        if (dim == 0 && count != 0)
        {
            throw Invalid($"dim 0 with {count} rows: no dimension committed");
        }

        int bytesPerPlane = dim / 8;
        int bytesPerPackedRow = checked(bytesPerPlane * bitWidth);
        byte[] packed = new byte[checked(count * bytesPerPackedRow)];
        float[] scales = new float[count];
        ulong[] ids = kind == 1 ? new ulong[count] : [];
        int completeBlocks = count / Block;
        int nByteGroups = dim == 0 ? 0 : dim / (8 / bitWidth);
        int planeSignBytes = bitWidth == 2 ? dim / 8 : bitWidth == 4 ? dim / 8 : 0;
        int planeLowRowBytes = bitWidth == 2 ? dim / 8 : bitWidth == 4 ? 3 * dim / 8 : 0;

        for (int block = 0; block < completeBlocks; block++)
        {
            int at = checked(unitOffset + block * unitBytes);
            ReadOnlySpan<byte> unit = raw.Slice(at, unitBytes);
            int codeLength = checked(Block * rowBytes);
            for (int lane = 0; lane < Block; lane++)
            {
                int row = block * Block + lane;
                int scaleAt = codeLength + lane * sizeof(float);
                float scale = ReadSingle(unit, scaleAt);
                ValidateVectorScale(scale, $"block {block}");
                if (row < count)
                {
                    scales[row] = scale;
                    if (kind == 1)
                    {
                        ids[row] = ReadUInt64(unit, codeLength + Block * sizeof(float) + lane * sizeof(ulong));
                    }

                    ConvertUnitRowToPacked(unit[..codeLength], layout, bitWidth, dim, rowBytes,
                        nByteGroups, planeSignBytes, planeLowRowBytes, lane,
                        packed.AsSpan(row * bytesPerPackedRow, bytesPerPackedRow));
                }
            }
        }

        foreach (RedoOp op in selected.Operations)
        {
            int row = op.Row;
            ReadOnlySpan<byte> payload = raw.Slice(op.PayloadOffset, tailRowBytes);
            float scale = ReadSingle(payload, rowBytes);
            ValidateVectorScale(scale, "pending op");
            scales[row] = scale;
            if (kind == 1)
            {
                ids[row] = ReadUInt64(payload, rowBytes + sizeof(float));
            }

            ConvertRowToPacked(payload[..rowBytes], bitWidth, dim, rowBytes,
                packed.AsSpan(row * bytesPerPackedRow, bytesPerPackedRow));
        }

        int tailCount = count % Block;
        for (int lane = 0; lane < tailCount; lane++)
        {
            int row = completeBlocks * Block + lane;
            int at = checked(selected.TailOffset + lane * tailRowBytes);
            ReadOnlySpan<byte> tail = raw.Slice(at, tailRowBytes);
            float scale = ReadSingle(tail, rowBytes);
            ValidateVectorScale(scale, "commit tail");
            scales[row] = scale;
            if (kind == 1)
            {
                ids[row] = ReadUInt64(tail, rowBytes + sizeof(float));
            }

            ConvertRowToPacked(tail[..rowBytes], bitWidth, dim, rowBytes,
                packed.AsSpan(row * bytesPerPackedRow, bytesPerPackedRow));
        }

        return new LoadedIndex(bitWidth, dim, count, packed, scales, ids, shift, tqScale,
            nonce, selected.Generation, isV8, selected.AppendCompatible);
    }

    private static void WriteImage(
        Stream output,
        int kind,
        int bitWidth,
        int dim,
        int nVectors,
        ReadOnlySpan<byte> packedCodes,
        ReadOnlySpan<float> scales,
        ReadOnlySpan<ulong> ids,
        ReadOnlySpan<float> shift,
        ReadOnlySpan<float> tqScale,
        ulong nonce = 0)
    {
        const byte layout = SequentialLayout;
        int levels = 1 << bitWidth;
        float[] boundaries;
        float[] centroids;
        if (dim == 0)
        {
            boundaries = new float[levels - 1];
            centroids = new float[levels];
        }
        else
        {
            (boundaries, centroids) = Codebook.Compute(bitWidth, dim);
        }

        using var writer = new BinaryWriter(output, System.Text.Encoding.UTF8, leaveOpen: true);
        using var superblock = new MemoryStream();
        using (var sw = new BinaryWriter(superblock, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            sw.Write(V8Magic);
            sw.Write(V8Version);
            sw.Write((byte)bitWidth);
            sw.Write((byte)kind);
            sw.Write(layout);
            WriteUInt32(sw, checked((uint)dim));
            WriteUInt64(sw, nonce);
            WriteUInt32(sw, MaxOps);
            foreach (float value in boundaries)
            {
                WriteSingle(sw, value);
            }

            foreach (float value in centroids)
            {
                WriteSingle(sw, value);
            }

            WriteUInt32(sw, checked((uint)shift.Length));
            foreach (float value in shift)
            {
                WriteSingle(sw, value);
            }

            foreach (float value in tqScale)
            {
                WriteSingle(sw, value);
            }
        }

        byte[] superblockBytes = superblock.ToArray();
        writer.Write(superblockBytes);
        WriteUInt32(writer, Crc32(superblockBytes));

        int rowBytes = dim == 0 ? 0 : dim / (8 / bitWidth);
        int idBytes = kind == 1 ? sizeof(ulong) : 0;
        int tailRowBytes = rowBytes + sizeof(float) + idBytes;
        int headerBytes = HeaderLength(rowBytes, idBytes);
        int unitBytes = Block * (rowBytes + sizeof(float) + idBytes);
        int tailCount = nVectors % Block;
        byte[] header = CreateSnapshotHeader(0, nVectors, tailCount, rowBytes,
            bitWidth, dim, packedCodes, scales, ids, kind);
        writer.Write(header);
        WriteZeros(writer, headerBytes - header.Length);
        WriteZeros(writer, headerBytes);

        int bytesPerPlane = dim / 8;
        int packedRowBytes = bytesPerPlane * bitWidth;
        int completeBlocks = nVectors / Block;
        byte[] unit = new byte[unitBytes];
        for (int block = 0; block < completeBlocks; block++)
        {
            int codeLength = Block * rowBytes;
            Array.Clear(unit);
            for (int lane = 0; lane < Block; lane++)
            {
                int row = block * Block + lane;
                EncodePackedRow(packedCodes.Slice(row * packedRowBytes, packedRowBytes),
                    bitWidth, dim, unit.AsSpan(0, codeLength), block, lane, rowBytes);
                WriteSingle(unit.AsSpan(codeLength + lane * sizeof(float), sizeof(float)), scales[row]);
                if (kind == 1)
                {
                    WriteUInt64(unit.AsSpan(codeLength + Block * sizeof(float) + lane * sizeof(ulong), sizeof(ulong)), ids[row]);
                }
            }

            writer.Write(unit);
        }

        writer.Flush();
    }

    private static byte[] CreateSnapshotHeader(
        ulong generation,
        int nVectors,
        int tailCount,
        int rowBytes,
        int bitWidth,
        int dim,
        ReadOnlySpan<byte> packedCodes,
        ReadOnlySpan<float> scales,
        ReadOnlySpan<ulong> ids,
        int kind)
    {
        using var header = new MemoryStream();
        using var writer = new BinaryWriter(header);
        WriteUInt64(writer, generation);
        WriteUInt64(writer, checked((ulong)nVectors));
        int bytesPerPlane = dim / 8;
        int packedRowBytes = bytesPerPlane * bitWidth;
        int firstTail = nVectors - tailCount;
        byte[] row = new byte[rowBytes];
        for (int i = 0; i < tailCount; i++)
        {
            int index = firstTail + i;
            EncodePackedRow(packedCodes.Slice(index * packedRowBytes, packedRowBytes), bitWidth, dim, row);
            writer.Write(row);
            WriteSingle(writer, scales[index]);
            if (kind == 1)
            {
                WriteUInt64(writer, ids[index]);
            }
        }

        WriteUInt32(writer, 0); // no pending redo groups
        WriteUInt32(writer, 0); // no materialized units
        WriteUInt32(writer, 0);
        WriteUInt32(writer, 0);
        Span<byte> generationBytes = stackalloc byte[sizeof(ulong)];
        BinaryPrimitives.WriteUInt64LittleEndian(generationBytes, generation);
        WriteUInt32(writer, Crc32(generationBytes));
        byte[] body = header.ToArray();
        WriteUInt32(writer, Crc32(body));
        return header.ToArray();
    }

    private static ParsedHeader? ParseHeader(
        ReadOnlySpan<byte> raw,
        int headerOffset,
        int slot,
        int fileLength,
        int dim,
        int rowBytes,
        int tailRowBytes,
        int idBytes,
        int unitOffset,
        int unitBytes,
        int headerCapacity)
    {
        int slotEnd = Math.Min(raw.Length, checked(headerOffset + headerCapacity));
        raw = raw[..slotEnd];
        if (headerOffset < 0 || headerOffset > raw.Length - 16)
        {
            return null;
        }

        int p = headerOffset;
        ulong generation = ReadUInt64(raw, p);
        if ((generation & 1) != (uint)slot)
        {
            return null;
        }

        ulong count64 = ReadUInt64(raw, p + 8);
        if (count64 > int.MaxValue)
        {
            return null;
        }

        int count = (int)count64;
        int blocks = count / Block;
        long endUnits = (long)unitOffset + (long)blocks * unitBytes;
        if (endUnits > fileLength)
        {
            return null;
        }

        int tailBytes = checked((count % Block) * tailRowBytes);
        p = checked(p + 16 + tailBytes);
        if (p > raw.Length - 4)
        {
            return null;
        }

        uint groupCount = ReadUInt32(raw, p);
        if (groupCount > MaxOps)
        {
            return null;
        }

        p += 4;
        var operations = new List<RedoOp>();
        var touchedRows = new HashSet<int>();
        for (int group = 0; group < groupCount; group++)
        {
            if (p > raw.Length - 5)
            {
                return null;
            }

            uint block = ReadUInt32(raw, p);
            byte opCount = raw[p + 4];
            p += 5;
            if (block >= blocks || opCount == 0)
            {
                return null;
            }

            for (int op = 0; op < opCount; op++)
            {
                if (p >= raw.Length)
                {
                    return null;
                }

                int lane = raw[p];
                if (lane >= Block)
                {
                    return null;
                }

                int row = checked((int)block * Block + lane);
                if (row >= count || !touchedRows.Add(row))
                {
                    return null;
                }

                int payloadOffset = checked(p + 1);
                p = checked(p + 1 + rowBytes + sizeof(float) + idBytes);
                if (p > raw.Length)
                {
                    return null;
                }

                operations.Add(new RedoOp(row, payloadOffset));
            }
        }

        if (p > raw.Length - 4)
        {
            return null;
        }

        uint materializedCount = ReadUInt32(raw, p);
        if (materializedCount > MaxOps)
        {
            return null;
        }

        p += 4;
        var materialized = new List<int>((int)materializedCount);
        var materializedSet = new HashSet<int>();
        for (int i = 0; i < materializedCount; i++)
        {
            if (p > raw.Length - 4)
            {
                return null;
            }

            uint block = ReadUInt32(raw, p);
            if (block >= blocks || !materializedSet.Add((int)block))
            {
                return null;
            }

            materialized.Add((int)block);
            p += 4;
        }

        if (p > raw.Length - 16)
        {
            return null;
        }

        uint appFrom = ReadUInt32(raw, p);
        uint appTo = ReadUInt32(raw, p + 4);
        uint deltaCrc = ReadUInt32(raw, p + 8);
        p += 12;
        uint storedCrc = ReadUInt32(raw, p);
        if (Crc32(raw.Slice(headerOffset, p - headerOffset)) != storedCrc)
        {
            return null;
        }

        if (appFrom > appTo || appTo > blocks)
        {
            return null;
        }

        var deltaBlocks = new List<int>(materialized.Count + (int)(appTo - appFrom));
        deltaBlocks.AddRange(materialized);
        for (uint b = appFrom; b < appTo; b++)
        {
            if (!materializedSet.Add((int)b))
            {
                return null;
            }

            deltaBlocks.Add((int)b);
        }

        int tailOffset = headerOffset + 16;
        return new ParsedHeader(generation, count, tailOffset, operations, deltaBlocks, deltaCrc,
            p + sizeof(uint), groupCount == 0 && materializedCount == 0);
    }

    private static bool VerifyDelta(ReadOnlySpan<byte> raw, ParsedHeader header, int unitOffset, int unitBytes)
    {
        int total = checked(8 + header.DeltaBlocks.Count * (4 + unitBytes));
        var digest = new CrcAccumulator(total);
        Span<byte> number = stackalloc byte[sizeof(ulong)];
        Span<byte> blockBytes = stackalloc byte[sizeof(uint)];
        BinaryPrimitives.WriteUInt64LittleEndian(number, header.Generation);
        digest.Push(number);
        foreach (int block in header.DeltaBlocks)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(blockBytes, (uint)block);
            digest.Push(blockBytes);
            int at = checked(unitOffset + block * unitBytes);
            if (at < 0 || at > raw.Length - unitBytes)
            {
                return false;
            }

            digest.Push(raw.Slice(at, unitBytes));
        }

        return digest.Finish() == header.DeltaCrc;
    }

    private static void ConvertUnitRowToPacked(
        ReadOnlySpan<byte> codeData,
        byte layout,
        int bitWidth,
        int dim,
        int rowBytes,
        int nByteGroups,
        int signBytes,
        int lowRowBytes,
        int lane,
        Span<byte> destination)
    {
        if (layout == PlanesLayout)
        {
            int signBlockBytes = signBytes * Block;
            int lowBlockOffset = signBlockBytes;
            int lowAt = lowBlockOffset + lane * lowRowBytes;
            codeData.Slice(lowAt, lowRowBytes).CopyTo(destination);
            for (int group = 0; group < signBytes; group++)
            {
                destination[lowRowBytes + group] = codeData[group * Block + lane];
            }

            return;
        }

        Span<byte> rowCodes = stackalloc byte[rowBytes];
        for (int group = 0; group < rowBytes; group++)
        {
            rowCodes[group] = codeData[group * Block + lane];
        }

        ConvertRowToPacked(rowCodes, bitWidth, dim, rowBytes, destination);
    }

    private static void ConvertRowToPacked(
        ReadOnlySpan<byte> rowCodes,
        int bitWidth,
        int dim,
        int rowBytes,
        Span<byte> destination)
    {
        if (dim == 0)
        {
            return;
        }

        int codesPerByte = 8 / bitWidth;
        int fieldWidth = bitWidth == 3 ? 4 : bitWidth;
        int expectedGroups = dim / codesPerByte;
        if (rowCodes.Length != rowBytes || rowCodes.Length != expectedGroups)
        {
            throw Invalid("invalid code row length");
        }

        int bytesPerPlane = dim / 8;
        destination.Clear();
        for (int d = 0; d < dim; d++)
        {
            int group = d / codesPerByte;
            int within = d % codesPerByte;
            int shift = (codesPerByte - 1 - within) * fieldWidth;
            int code = (rowCodes[group] >> shift) & ((1 << bitWidth) - 1);
            int bitMask = 1 << (7 - (d & 7));
            for (int plane = 0; plane < bitWidth; plane++)
            {
                if ((code & (1 << plane)) != 0)
                {
                    destination[plane * bytesPerPlane + (d >> 3)] |= (byte)bitMask;
                }
            }
        }
    }

    private static void EncodePackedRow(
        ReadOnlySpan<byte> packedRow,
        int bitWidth,
        int dim,
        Span<byte> destination,
        int block = -1,
        int lane = -1,
        int rowBytes = -1)
    {
        int bytesPerPlane = dim / 8;
        int groups = dim == 0 ? 0 : dim / (8 / bitWidth);
        int bytesPerPackedRow = checked(bytesPerPlane * bitWidth);
        if (packedRow.Length != bytesPerPackedRow || destination.Length < groups)
        {
            throw new ArgumentException("Packed row or destination has an invalid length.");
        }

        int fieldWidth = bitWidth == 3 ? 4 : bitWidth;
        int codesPerByte = 8 / bitWidth;
        Span<byte> row = block >= 0 ? stackalloc byte[groups] : destination[..groups];
        row.Clear();
        for (int d = 0; d < dim; d++)
        {
            int code = 0;
            int bit = 7 - (d & 7);
            for (int plane = 0; plane < bitWidth; plane++)
            {
                if ((packedRow[plane * bytesPerPlane + (d >> 3)] & (1 << bit)) != 0)
                {
                    code |= 1 << plane;
                }
            }

            int group = d / codesPerByte;
            int shift = (codesPerByte - 1 - (d % codesPerByte)) * fieldWidth;
            row[group] |= (byte)(code << shift);
        }

        if (block >= 0)
        {
            for (int group = 0; group < groups; group++)
            {
                destination[group * Block + lane] = row[group];
            }
        }
    }

    private static byte[] CreateRow(ReadOnlySpan<byte> packedRow, int bitWidth, int dim)
    {
        int groups = dim == 0 ? 0 : dim / (8 / bitWidth);
        byte[] row = new byte[groups];
        EncodePackedRow(packedRow, bitWidth, dim, row);
        return row;
    }

    private static void Validate(
        int kind,
        int bitWidth,
        int dim,
        int nVectors,
        ReadOnlySpan<byte> packedCodes,
        ReadOnlySpan<float> scales,
        ReadOnlySpan<ulong> ids,
        ReadOnlySpan<float> shift,
        ReadOnlySpan<float> tqScale)
    {
        if (kind is not (0 or 1))
        {
            throw new ArgumentOutOfRangeException(nameof(kind));
        }

        InputValidation.ValidateBitWidth(bitWidth, nameof(bitWidth));
        if (dim == 0)
        {
            if (nVectors != 0)
            {
                throw new ArgumentOutOfRangeException(nameof(nVectors), "nVectors must be zero when dim is zero.");
            }
        }
        else
        {
            InputValidation.ValidateDimension(dim, nameof(dim));
        }

        ArgumentOutOfRangeException.ThrowIfNegative(nVectors);
        int expectedCodeBytes = checked(dim / 8 * bitWidth * nVectors);
        if (packedCodes.Length != expectedCodeBytes || scales.Length != nVectors)
        {
            throw new ArgumentException("Packed code or scale length does not match the index geometry.");
        }

        if ((kind == 1 && ids.Length != nVectors) || (kind == 0 && !ids.IsEmpty))
        {
            throw new ArgumentException("Id array length does not match the index kind and vector count.", nameof(ids));
        }

        if (shift.Length != tqScale.Length || (shift.Length != 0 && shift.Length != dim))
        {
            throw new ArgumentException("TQ+ calibration arrays must both be empty or have one value per dimension.");
        }

        ValidateCalibration(shift, tqScale);
        for (int i = 0; i < scales.Length; i++)
        {
            ValidateVectorScale(scales[i], $"vector {i}");
        }
    }

    private static void ValidateCalibration(ReadOnlySpan<float> shift, ReadOnlySpan<float> scale)
    {
        if (shift.Length != scale.Length)
        {
            throw Invalid("TQ+ calibration arrays have different lengths");
        }

        float maxShift = float.MaxValue / (Math.Max(1, shift.Length) * Constants.MaxInputMagnitude);
        maxShift /= 10.0f;
        float minScale = (Math.Max(1, scale.Length) * Constants.MaxInputMagnitude) / float.MaxValue;
        minScale *= 10.0f;
        for (int i = 0; i < shift.Length; i++)
        {
            if (!float.IsFinite(shift[i]) || MathF.Abs(shift[i]) > maxShift)
            {
                throw Invalid($"invalid TQ+ shift at coordinate {i}");
            }

            if (!float.IsFinite(scale[i]) || scale[i] < minScale)
            {
                throw Invalid($"invalid TQ+ scale at coordinate {i}");
            }
        }
    }

    private static void ValidateVectorScale(float scale, string location)
    {
        if (!float.IsFinite(scale) || scale < 0.0f || scale > MaxVectorScale)
        {
            throw Invalid($"invalid per-vector scale in {location}");
        }
    }

    private static bool PlanesGeometryValid(int bits, int groups)
        => bits switch
        {
            2 => groups % 8 == 0,
            4 => groups % 16 == 0,
            _ => false,
        };

    private static int HeaderLength(int rowBytes, int idBytes)
    {
        int opSize = checked(1 + rowBytes + sizeof(float) + idBytes);
        return checked(16 + 31 * (rowBytes + sizeof(float) + idBytes)
            + 4 + MaxOps * (5 + opSize) + 4 + MaxOps * 4 + 12 + 4);
    }

    private static int GetImageLength(int dim, int bits, int rows, int kind, int calibrationLength)
    {
        int fixedBytes = 24;
        int levels = 1 << bits;
        int rowBytes = dim == 0 ? 0 : dim / (8 / bits);
        int idBytes = kind == 1 ? sizeof(ulong) : 0;
        int superblock = checked(fixedBytes + (2 * levels - 1) * sizeof(float) + sizeof(uint)
            + 2 * calibrationLength * sizeof(float) + sizeof(uint));
        return checked(superblock + 2 * HeaderLength(rowBytes, idBytes)
            + (rows / Block) * Block * (rowBytes + sizeof(float) + idBytes));
    }

    private static void WriteZeros(BinaryWriter writer, int count)
    {
        byte[] zeros = new byte[Math.Min(count, 64 * 1024)];
        while (count > 0)
        {
            int take = Math.Min(count, zeros.Length);
            writer.Write(zeros, 0, take);
            count -= take;
        }
    }

    private static uint Crc32(ReadOnlySpan<byte> data)
    {
        if (data.Length < 4096)
        {
            return ~UpdateCrc(0xffff_ffff, data);
        }

        int third = data.Length / 3;
        uint c0 = ~UpdateCrc(0xffff_ffff, data[..third]);
        uint c1 = ~UpdateCrc(0xffff_ffff, data.Slice(third, third));
        uint c2 = ~UpdateCrc(0xffff_ffff, data[(third * 2)..]);
        Span<byte> digests = stackalloc byte[12];
        BinaryPrimitives.WriteUInt32LittleEndian(digests, c0);
        BinaryPrimitives.WriteUInt32LittleEndian(digests[4..], c1);
        BinaryPrimitives.WriteUInt32LittleEndian(digests[8..], c2);
        return ~UpdateCrc(0xffff_ffff, digests);
    }

    private static uint UpdateCrc(uint crc, ReadOnlySpan<byte> data)
    {
        foreach (byte value in data)
        {
            crc ^= value;
            for (int bit = 0; bit < 8; bit++)
            {
                crc = (crc >> 1) ^ ((crc & 1) == 0 ? 0 : 0x82f6_3b78u);
            }
        }

        return crc;
    }

    private static uint ReadUInt32(ReadOnlySpan<byte> bytes, int offset)
        => BinaryPrimitives.ReadUInt32LittleEndian(bytes.Slice(offset, sizeof(uint)));

    private static ulong ReadUInt64(ReadOnlySpan<byte> bytes, int offset)
        => BinaryPrimitives.ReadUInt64LittleEndian(bytes.Slice(offset, sizeof(ulong)));

    private static float ReadSingle(ReadOnlySpan<byte> bytes, int offset)
        => BitConverter.UInt32BitsToSingle(ReadUInt32(bytes, offset));

    private static void WriteUInt32(BinaryWriter writer, uint value)
    {
        Span<byte> bytes = stackalloc byte[sizeof(uint)];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
        writer.Write(bytes);
    }

    private static void WriteUInt64(BinaryWriter writer, ulong value)
    {
        Span<byte> bytes = stackalloc byte[sizeof(ulong)];
        BinaryPrimitives.WriteUInt64LittleEndian(bytes, value);
        writer.Write(bytes);
    }

    private static void WriteSingle(BinaryWriter writer, float value)
        => WriteUInt32(writer, BitConverter.SingleToUInt32Bits(value));

    private static void WriteSingle(Span<byte> destination, float value)
        => BinaryPrimitives.WriteUInt32LittleEndian(destination, BitConverter.SingleToUInt32Bits(value));

    private static void WriteUInt64(Span<byte> destination, ulong value)
        => BinaryPrimitives.WriteUInt64LittleEndian(destination, value);

    private static InvalidDataException Invalid(string message) => new(message);

    private sealed record RedoOp(int Row, int PayloadOffset);
    private sealed record ParsedHeader(
        ulong Generation,
        int Count,
        int TailOffset,
        List<RedoOp> Operations,
        List<int> DeltaBlocks,
        uint DeltaCrc,
        int UsedBytes,
        bool AppendCompatible);
    private sealed record ParsedSyncHeader(ulong Generation, int Count, bool AppendCompatible);

    private sealed class CrcAccumulator
    {
        private readonly int _split1;
        private readonly int _split2;
        private readonly bool _split;
        private readonly uint[] _parts = [0xffff_ffff, 0xffff_ffff, 0xffff_ffff];
        private int _position;

        internal CrcAccumulator(int totalLength)
        {
            _split = totalLength >= 4096;
            int third = totalLength / 3;
            _split1 = third;
            _split2 = third * 2;
        }

        internal void Push(ReadOnlySpan<byte> bytes)
        {
            if (!_split)
            {
                _parts[0] = UpdateCrc(_parts[0], bytes);
                return;
            }

            while (!bytes.IsEmpty)
            {
                int part = _position < _split1 ? 0 : _position < _split2 ? 1 : 2;
                int end = part == 0 ? _split1 : part == 1 ? _split2 : int.MaxValue;
                int take = Math.Min(bytes.Length, end - _position);
                _parts[part] = UpdateCrc(_parts[part], bytes[..take]);
                _position += take;
                bytes = bytes[take..];
            }
        }

        internal uint Finish()
        {
            if (!_split)
            {
                return ~_parts[0];
            }

            Span<byte> digests = stackalloc byte[12];
            BinaryPrimitives.WriteUInt32LittleEndian(digests, ~_parts[0]);
            BinaryPrimitives.WriteUInt32LittleEndian(digests[4..], ~_parts[1]);
            BinaryPrimitives.WriteUInt32LittleEndian(digests[8..], ~_parts[2]);
            return Crc32(digests);
        }
    }
}
