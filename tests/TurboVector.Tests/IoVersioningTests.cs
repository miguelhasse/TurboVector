using System.Buffers.Binary;

namespace TurboVector.Tests;

public class IoVersioningTests
{
    [Fact]
    public void RustV8SnapshotRoundTripsRowsAcrossUnitsAndTail()
    {
        const int dim = 64;
        const int nVectors = 67;
        foreach (int bitWidth in new[] { 2, 3, 4 })
        {
            float[] vectors = TestHelpers.MakeVectors(nVectors, dim, seed: 230UL + (ulong)bitWidth);
            var index = new TurboQuantIndex(dim, bitWidth);
            index.Add(vectors);

            byte[] image = index.ToBytes();
            Assert.Equal("TV8\0"u8.ToArray(), image.AsSpan(0, 4).ToArray());
            Assert.Equal((byte)1, image[4]);
            Assert.Equal(image, index.ToBytes());

            TurboQuantIndex loaded = TurboQuantIndex.Load(image);
            Assert.Equal(index.PackedCodes.ToArray(), loaded.PackedCodes.ToArray());
            Assert.Equal(index.Scales.ToArray(), loaded.Scales.ToArray());
            Assert.Equal(index.Search(vectors.AsSpan(0, 2 * dim), 8).Indices,
                loaded.Search(vectors.AsSpan(0, 2 * dim), 8).Indices);

            using var output = new MemoryStream();
            index.WriteTo(output);
            Assert.Equal(image, output.ToArray());
            output.Position = 0;
            TurboQuantIndex streamLoaded = TurboQuantIndex.Load(output);
            Assert.Equal(index.PackedCodes.ToArray(), streamLoaded.PackedCodes.ToArray());
        }
    }

    [Fact]
    public void RustV7SnapshotLoadsAndMatchesV8Data()
    {
        const int dim = 64;
        const int bitWidth = 3;
        float[] vectors = TestHelpers.MakeVectors(45, dim, seed: 81);
        var index = new TurboQuantIndex(dim, bitWidth);
        index.Add(vectors);

        TurboQuantIndex loaded = TurboQuantIndex.Load(ConvertSnapshotToV7(index.ToBytes(), bitWidth));

        Assert.Equal(index.PackedCodes.ToArray(), loaded.PackedCodes.ToArray());
        Assert.Equal(index.Scales.ToArray(), loaded.Scales.ToArray());
        Assert.Equal(index.Search(vectors.AsSpan(0, dim), 10).Indices,
            loaded.Search(vectors.AsSpan(0, dim), 10).Indices);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void RustV8PlanesLayoutLoads(int bitWidth)
    {
        const int dim = 256;
        float[] vectors = TestHelpers.MakeVectors(64, dim, seed: 810UL + (ulong)bitWidth);
        var index = new TurboQuantIndex(dim, bitWidth);
        index.Add(vectors);

        byte[] planes = ConvertSnapshotToPlanes(index.ToBytes(), bitWidth, dim, 64);
        TurboQuantIndex loaded = TurboQuantIndex.Load(planes);

        Assert.Equal(index.PackedCodes.ToArray(), loaded.PackedCodes.ToArray());
        Assert.Equal(index.Scales.ToArray(), loaded.Scales.ToArray());
        Assert.Equal(index.Search(vectors.AsSpan(0, 3 * dim), 10).Indices,
            loaded.Search(vectors.AsSpan(0, 3 * dim), 10).Indices);
    }

    [Fact]
    public void RawV8AndIdMapFilesRoundTrip()
    {
        string path = TestHelpers.TempPath(".tv", "v8-roundtrip");
        string idPath = TestHelpers.TempPath(".tvim", "v8-id-roundtrip");
        const int bitWidth = 4;
        const int dim = 32;
        const int nVectors = 3;
        byte[] packed = Enumerable.Repeat((byte)0xAB, (dim / 8) * bitWidth * nVectors).ToArray();
        float[] scales = [1.5f, 2.5f, 3.5f];
        ulong[] ids = [100UL, 200UL, 300UL];

        try
        {
            Io.Write(path, bitWidth, dim, nVectors, packed, scales);
            var (bw, d, n, p, s) = Io.Load(path);
            Assert.Equal(bitWidth, bw);
            Assert.Equal(dim, d);
            Assert.Equal(nVectors, n);
            Assert.Equal(packed, p);
            Assert.Equal(scales, s);

            Io.WriteIdMap(idPath, bitWidth, dim, nVectors, packed, scales, ids);
            var (idBw, idDim, idCount, idPacked, idScales, slotIds) = Io.LoadIdMap(idPath);
            Assert.Equal(bitWidth, idBw);
            Assert.Equal(dim, idDim);
            Assert.Equal(nVectors, idCount);
            Assert.Equal(packed, idPacked);
            Assert.Equal(scales, idScales);
            Assert.Equal(ids, slotIds);
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
            TestHelpers.DeleteFileIfExists(idPath);
        }
    }

    [Fact]
    public void InvalidWritesPreserveExistingFiles()
    {
        string path = TestHelpers.TempPath(".tv", "invalid-overwrite");
        byte[] original = "keep-existing-index"u8.ToArray();
        try
        {
            File.WriteAllBytes(path, original);
            Assert.Throws<ArgumentException>(() => Io.Write(path, 4, 32, 1, Array.Empty<byte>(), [1.0f]));
            Assert.Equal(original, File.ReadAllBytes(path));
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
        }
    }

    [Fact]
    public void PathWriteReplacesExistingIndex()
    {
        string path = TestHelpers.TempPath(".tv", "overwrite");
        try
        {
            Io.Write(path, 2, 32, 1, new byte[8], [1.0f]);
            Io.Write(path, 4, 32, 1, new byte[16], [2.0f]);
            var (bits, dim, count, packed, scales) = Io.Load(path);
            Assert.Equal(4, bits);
            Assert.Equal(32, dim);
            Assert.Equal(1, count);
            Assert.Equal(new byte[16], packed);
            Assert.Equal([2.0f], scales);
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
        }
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void SyncAppendsFullUnitsAndPreservesTailRows(int bitWidth)
    {
        const int dim = 64;
        string path = TestHelpers.TempPath(".tv", $"sync-append-{bitWidth}");
        float[] initial = TestHelpers.MakeVectors(3, dim, seed: 70 + (ulong)bitWidth);
        float[] firstAppend = TestHelpers.MakeVectors(29, dim, seed: 80 + (ulong)bitWidth);
        float[] tailAppend = TestHelpers.MakeVectors(2, dim, seed: 90 + (ulong)bitWidth);
        var index = new TurboQuantIndex(dim, bitWidth);
        try
        {
            index.Add(initial);
            index.Sync(path);
            long initialLength = new FileInfo(path).Length;

            index.Add(firstAppend);
            index.Sync(path);
            Assert.Equal(initialLength + 32L * (dim / (8 / bitWidth) + sizeof(float)), new FileInfo(path).Length);

            index.Add(tailAppend);
            index.Sync(path);
            Assert.Equal(initialLength + 32L * (dim / (8 / bitWidth) + sizeof(float)), new FileInfo(path).Length);

            TurboQuantIndex loaded = TurboQuantIndex.Load(path);
            Assert.Equal(index.PackedCodes.ToArray(), loaded.PackedCodes.ToArray());
            Assert.Equal(index.Scales.ToArray(), loaded.Scales.ToArray());

            loaded.Add(TestHelpers.MakeVectors(30, dim, seed: 100 + (ulong)bitWidth));
            loaded.Sync(path);
            TurboQuantIndex reloaded = TurboQuantIndex.Load(path);
            Assert.Equal(64, reloaded.Len);
            Assert.Equal(loaded.PackedCodes.ToArray(), reloaded.PackedCodes.ToArray());
            Assert.Equal(loaded.Scales.ToArray(), reloaded.Scales.ToArray());
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
        }
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void SyncCanAppendBeforeTheFirstFullBlock(int bitWidth)
    {
        const int dim = 64;
        string path = TestHelpers.TempPath(".tv", $"sync-first-tail-{bitWidth}");
        var index = new TurboQuantIndex(dim, bitWidth);
        try
        {
            index.Add(TestHelpers.MakeVectors(1, dim, seed: 201 + (ulong)bitWidth));
            index.Sync(path);
            index.Add(TestHelpers.MakeVectors(1, dim, seed: 211 + (ulong)bitWidth));
            index.Sync(path);

            TurboQuantIndex loaded = TurboQuantIndex.Load(path);
            Assert.Equal(2, loaded.Len);
            Assert.Equal(index.PackedCodes.ToArray(), loaded.PackedCodes.ToArray());
            Assert.Equal(index.Scales.ToArray(), loaded.Scales.ToArray());
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
        }
    }

    [Fact]
    public void SyncRejectsAPathReplacedByAnotherWriter()
    {
        const int dim = 32;
        string path = TestHelpers.TempPath(".tv", "sync-foreign-writer");
        var first = new TurboQuantIndex(dim, 4);
        var second = new TurboQuantIndex(dim, 4);
        first.Add(TestHelpers.MakeVectors(3, dim, seed: 501));
        second.Add(TestHelpers.MakeVectors(5, dim, seed: 502));
        try
        {
            first.Sync(path);
            second.Sync(path);
            byte[] foreignImage = File.ReadAllBytes(path);
            first.Add(TestHelpers.MakeVectors(1, dim, seed: 503));

            InvalidDataException error = Assert.Throws<InvalidDataException>(() => first.Sync(path));

            Assert.Contains("no longer matches", error.Message);
            Assert.Equal(foreignImage, File.ReadAllBytes(path));
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
        }
    }

    [Fact]
    public void SnapshotWriteToBoundPathAllowsSyncToReclaimIt()
    {
        const int dim = 64;
        string path = TestHelpers.TempPath(".tv", "sync-after-write");
        var index = new TurboQuantIndex(dim, 4);
        index.Add(TestHelpers.MakeVectors(3, dim, seed: 551));
        try
        {
            index.Sync(path);
            index.Add(TestHelpers.MakeVectors(4, dim, seed: 552));
            index.Write(path);
            index.Add(TestHelpers.MakeVectors(29, dim, seed: 553));
            index.Sync(path);

            TurboQuantIndex loaded = TurboQuantIndex.Load(path);
            Assert.Equal(36, loaded.Len);
            Assert.Equal(index.PackedCodes.ToArray(), loaded.PackedCodes.ToArray());
            Assert.Equal(index.Scales.ToArray(), loaded.Scales.ToArray());
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
        }
    }

    [Fact]
    public void SyncRecognizesWindowsCaseInsensitivePathAliases()
    {
        if (!OperatingSystem.IsWindows())
        {
            return;
        }

        const int dim = 64;
        string path = TestHelpers.TempPath(".tv", "sync-path-case");
        var index = new TurboQuantIndex(dim, 4);
        index.Add(TestHelpers.MakeVectors(3, dim, seed: 561));
        try
        {
            index.Sync(path);
            long snapshotLength = new FileInfo(path).Length;
            index.Add(TestHelpers.MakeVectors(29, dim, seed: 562));
            index.Sync(path.ToUpperInvariant());

            Assert.Equal(snapshotLength + 32L * (dim / 2 + sizeof(float)), new FileInfo(path).Length);
            Assert.Equal(index.PackedCodes.ToArray(), TurboQuantIndex.Load(path).PackedCodes.ToArray());
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
        }
    }

    [Fact]
    public void SyncCompactsAfterSwapRemove()
    {
        const int dim = 64;
        string path = TestHelpers.TempPath(".tv", "sync-swap-remove");
        var index = new TurboQuantIndex(dim, 2);
        index.Add(TestHelpers.MakeVectors(40, dim, seed: 601));
        try
        {
            index.Sync(path);
            index.SwapRemove(3);
            index.Sync(path);

            TurboQuantIndex loaded = TurboQuantIndex.Load(path);
            Assert.Equal(39, loaded.Len);
            Assert.Equal(index.PackedCodes.ToArray(), loaded.PackedCodes.ToArray());
            Assert.Equal(index.Scales.ToArray(), loaded.Scales.ToArray());
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
        }
    }

    [Fact]
    public void SyncRecoversFromOrphanedAppendedUnits()
    {
        const int dim = 64;
        const int bitWidth = 4;
        string path = TestHelpers.TempPath(".tv", "sync-orphaned-unit");
        var index = new TurboQuantIndex(dim, bitWidth);
        index.Add(TestHelpers.MakeVectors(3, dim, seed: 701));
        try
        {
            index.Sync(path);
            long committedLength = new FileInfo(path).Length;
            using (var stream = new FileStream(path, FileMode.Open, FileAccess.Write, FileShare.None))
            {
                stream.SetLength(committedLength + 32L * (dim / (8 / bitWidth) + sizeof(float)));
            }

            TurboQuantIndex loaded = TurboQuantIndex.Load(path);
            loaded.Add(TestHelpers.MakeVectors(29, dim, seed: 702));
            loaded.Sync(path);

            TurboQuantIndex recovered = TurboQuantIndex.Load(path);
            Assert.Equal(32, recovered.Len);
            Assert.Equal(loaded.PackedCodes.ToArray(), recovered.PackedCodes.ToArray());
            Assert.Equal(loaded.Scales.ToArray(), recovered.Scales.ToArray());
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
        }
    }

    [Fact]
    public void SyncRecoversWhenNewCommitHeaderIsTorn()
    {
        const int dim = 64;
        const int bitWidth = 2;
        string path = TestHelpers.TempPath(".tv", "sync-torn-header");
        var index = new TurboQuantIndex(dim, bitWidth);
        index.Add(TestHelpers.MakeVectors(3, dim, seed: 751));
        try
        {
            index.Sync(path);
            index.Add(TestHelpers.MakeVectors(29, dim, seed: 752));
            index.Sync(path);

            byte[] torn = File.ReadAllBytes(path);
            int levels = 1 << bitWidth;
            int rowBytes = dim / (8 / bitWidth);
            int headerBytes = 16 + 31 * (rowBytes + sizeof(float)) + 4
                + 1024 * (5 + 1 + rowBytes + sizeof(float)) + 4 + 1024 * sizeof(uint) + 12 + sizeof(uint);
            int superblockBytes = 24 + (2 * levels - 1) * sizeof(float) + 2 * sizeof(uint);
            torn[superblockBytes + headerBytes] ^= 0x40;
            File.WriteAllBytes(path, torn);

            TurboQuantIndex recoveredPrevious = TurboQuantIndex.Load(path);
            Assert.Equal(3, recoveredPrevious.Len);
            recoveredPrevious.Add(TestHelpers.MakeVectors(29, dim, seed: 753));
            recoveredPrevious.Sync(path);

            TurboQuantIndex recoveredLatest = TurboQuantIndex.Load(path);
            Assert.Equal(32, recoveredLatest.Len);
            Assert.Equal(recoveredPrevious.PackedCodes.ToArray(), recoveredLatest.PackedCodes.ToArray());
            Assert.Equal(recoveredPrevious.Scales.ToArray(), recoveredLatest.Scales.ToArray());
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
        }
    }

    [Fact]
    public void IdMapSyncPersistsIdsAndSupportsAppend()
    {
        const int dim = 32;
        string path = TestHelpers.TempPath(".tvim", "idmap-sync");
        var index = new IdMapIndex(dim, 2);
        index.AddWithIds(TestHelpers.MakeVectors(2, dim, seed: 801), [11UL, 22UL]);
        try
        {
            index.Sync(path);
            index.AddWithIds(TestHelpers.MakeVectors(31, dim, seed: 802), Enumerable.Range(0, 31).Select(static i => (ulong)(100 + i)).ToArray());
            index.Sync(path);

            IdMapIndex loaded = IdMapIndex.Load(path);
            Assert.Equal(33, loaded.Len);
            Assert.True(loaded.Contains(11));
            Assert.True(loaded.Contains(22));
            Assert.True(loaded.Contains(130));
            Assert.Equal(index.Search(TestHelpers.MakeVectors(1, dim, seed: 803), 10).Ids,
                loaded.Search(TestHelpers.MakeVectors(1, dim, seed: 803), 10).Ids);
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
        }
    }

    [Fact]
    public void SyncPreservesCalibrationAndRewritesAfterRecalibration()
    {
        const int dim = 64;
        string path = TestHelpers.TempPath(".tv", "sync-calibration");
        float[] sampleA = TestHelpers.MakeVectors(40, dim, seed: 901);
        float[] sampleB = TestHelpers.MakeVectors(40, dim, seed: 902);
        var index = new TurboQuantIndex(dim, 4);
        try
        {
            index.Calibrate(sampleA);
            index.Add(sampleA.AsSpan(0, 4 * dim));
            index.Sync(path);
            index.Add(sampleA.AsSpan(4 * dim, 30 * dim));
            index.Sync(path);

            index.Calibrate(sampleB);
            index.Sync(path);
            TurboQuantIndex loaded = TurboQuantIndex.Load(path);

            Assert.Equal(index.TqPlusShift.ToArray(), loaded.TqPlusShift.ToArray());
            Assert.Equal(index.TqPlusScale.ToArray(), loaded.TqPlusScale.ToArray());
            Assert.Equal(index.PackedCodes.ToArray(), loaded.PackedCodes.ToArray());
            Assert.Equal(index.Scales.ToArray(), loaded.Scales.ToArray());
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
        }
    }

    [Fact]
    public void LegacyCSharpV2FilesAreRejected()
    {
        string path = TestHelpers.TempPath(".tv", "v2-rejected");
        try
        {
            File.WriteAllBytes(path, [.. "TVPI"u8, 2, .. new byte[8]]);
            InvalidDataException error = Assert.Throws<InvalidDataException>(() => TurboQuantIndex.Load(path));
            Assert.Contains("legacy index format version 2", error.Message);
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
        }
    }

    [Fact]
    public void LegacyRustV1FilesGetRebuildHint()
    {
        string path = TestHelpers.TempPath(".tv", "v1-rejected");
        try
        {
            using (var writer = new BinaryWriter(File.Create(path)))
            {
                writer.Write((byte)4);
                writer.Write(32u);
                writer.Write(2u);
                writer.Write(0u);
            }

            InvalidDataException error = Assert.Throws<InvalidDataException>(() => Io.Load(path));
            Assert.Contains("turbovec ≤ 0.4.3", error.Message);
            Assert.Contains("Rebuild", error.Message);
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
        }
    }

    [Fact]
    public void UnknownMagicIsRejected()
    {
        byte[] garbage = [.. "NOPE"u8, .. new byte[32]];
        InvalidDataException error = Assert.Throws<InvalidDataException>(() => Io.LoadData(garbage));
        Assert.Contains("wrong magic", error.Message);
        Assert.DoesNotContain("turbovec ≤ 0.4.3", error.Message);
    }

    private static byte[] ConvertSnapshotToV7(byte[] v8, int bitWidth)
    {
        int levels = 1 << bitWidth;
        int calibrationLengthOffset = checked(24 + (2 * levels - 1) * sizeof(float));
        int calibrationLength = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(v8.AsSpan(calibrationLengthOffset)));
        int superblockLength = checked(calibrationLengthOffset + sizeof(uint)
            + 2 * calibrationLength * sizeof(float) + sizeof(uint) - 1);
        byte[] v7 = new byte[v8.Length - 1];
        v8.AsSpan(0, 7).CopyTo(v7);
        v8.AsSpan(8).CopyTo(v7.AsSpan(7));
        "TV7\0"u8.CopyTo(v7);
        v7[4] = 2;
        uint crc = Crc32C(v7.AsSpan(0, superblockLength - 4));
        BinaryPrimitives.WriteUInt32LittleEndian(v7.AsSpan(superblockLength - 4), crc);
        return v7;
    }

    private static byte[] ConvertSnapshotToPlanes(byte[] image, int bits, int dim, int rows)
    {
        byte[] result = image.ToArray();
        int levels = 1 << bits;
        int superblockLength = checked(24 + (2 * levels - 1) * sizeof(float) + 2 * sizeof(uint));
        int rowBytes = dim / (8 / bits);
        int unitBytes = 32 * (rowBytes + sizeof(float));
        int headerBytes = checked(16 + 31 * (rowBytes + sizeof(float)) + 4
            + 1024 * (5 + 1 + rowBytes + sizeof(float)) + 4 + 1024 * 4 + 12 + 4);
        int unitOffset = superblockLength + 2 * headerBytes;
        int bytesPerPlane = dim / 8;
        int groups = dim / (8 / bits);
        int signBytes = bytesPerPlane;
        int lowRowBytes = (bits - 1) * bytesPerPlane;
        int packedRowBytes = bits * bytesPerPlane;
        int fieldWidth = bits == 3 ? 4 : bits;
        int codesPerByte = 8 / bits;

        result[7] = 1;
        for (int block = 0; block < rows / 32; block++)
        {
            int at = unitOffset + block * unitBytes;
            byte[] seq = result.AsSpan(at, 32 * rowBytes).ToArray();
            byte[] converted = new byte[seq.Length];
            for (int lane = 0; lane < 32; lane++)
            {
                byte[] packed = new byte[packedRowBytes];
                for (int d = 0; d < dim; d++)
                {
                    int codeByte = seq[(d / codesPerByte) * 32 + lane];
                    int shift = (codesPerByte - 1 - d % codesPerByte) * fieldWidth;
                    int code = (codeByte >> shift) & ((1 << bits) - 1);
                    for (int plane = 0; plane < bits; plane++)
                    {
                        if ((code & (1 << plane)) != 0)
                        {
                            packed[plane * bytesPerPlane + d / 8] |= (byte)(1 << (7 - d % 8));
                        }
                    }
                }

                int signOffset = signBytes * 32;
                int lowAt = signOffset + lane * lowRowBytes;
                packed.AsSpan(0, lowRowBytes).CopyTo(converted.AsSpan(lowAt, lowRowBytes));
                for (int g = 0; g < signBytes; g++)
                {
                    converted[g * 32 + lane] = packed[(bits - 1) * bytesPerPlane + g];
                }
            }

            converted.CopyTo(result, at);
        }

        uint crc = Crc32C(result.AsSpan(0, superblockLength - sizeof(uint)));
        BinaryPrimitives.WriteUInt32LittleEndian(result.AsSpan(superblockLength - sizeof(uint)), crc);
        return result;
    }

    private static uint Crc32C(ReadOnlySpan<byte> bytes)
    {
        uint crc = 0xffff_ffff;
        foreach (byte value in bytes)
        {
            crc ^= value;
            for (int bit = 0; bit < 8; bit++)
            {
                crc = (crc >> 1) ^ ((crc & 1) == 0 ? 0 : 0x82f6_3b78u);
            }
        }

        return ~crc;
    }
}
