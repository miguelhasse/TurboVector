using System;
using System.IO;
using TurboVector;

namespace TurboVector.Tests;

public class IoVersioningTests
{
    [Fact]
    public void TvRoundTripCurrentFormat()
    {
        string path = TestHelpers.TempPath(".tv", "tv-v2");
        const int bitWidth = 4;
        const int dim = 32;
        const int nVectors = 3;
        byte[] packed = Enumerable.Repeat((byte)0xAB, (dim / 8) * bitWidth * nVectors).ToArray();
        float[] scales = [1.5f, 2.5f, 3.5f];

        try
        {
            Io.Write(path, bitWidth, dim, nVectors, packed, scales);
            var (bw, d, n, p, s) = Io.Load(path);
            Assert.Equal(bitWidth, bw);
            Assert.Equal(dim, d);
            Assert.Equal(nVectors, n);
            Assert.Equal(packed, p);
            Assert.Equal(scales, s);
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
        }
    }

    [Fact]
    public void TvV1FileIsRejectedWithUpgradeHint()
    {
        string path = TestHelpers.TempPath(".tv", "tv-v1");
        try
        {
            using (var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None))
            using (var writer = new BinaryWriter(stream))
            {
                writer.Write((byte)4);
                writer.Write(32u);
                writer.Write(2u);
                writer.Write(new byte[(32 / 8) * 4 * 2]);
                writer.Write(1.0f);
                writer.Write(2.0f);
            }

            Exception ex = Assert.ThrowsAny<Exception>(() => Io.Load(path));
            Assert.Contains("turbovec ≤ 0.4.3", ex.Message);
            Assert.Contains("Rebuild", ex.Message);
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
        }
    }

    [Fact]
    public void TvimRoundTripCurrentFormat()
    {
        string path = TestHelpers.TempPath(".tvim", "tvim-v2");
        const int bitWidth = 2;
        const int dim = 16;
        const int nVectors = 4;
        byte[] packed = Enumerable.Repeat((byte)0x55, (dim / 8) * bitWidth * nVectors).ToArray();
        float[] scales = [0.5f, 1.0f, 1.5f, 2.0f];
        ulong[] ids = [100UL, 200UL, 300UL, 400UL];

        try
        {
            Io.WriteIdMap(path, bitWidth, dim, nVectors, packed, scales, ids);
            var (bw, d, n, p, s, slotToId) = Io.LoadIdMap(path);
            Assert.Equal(bitWidth, bw);
            Assert.Equal(dim, d);
            Assert.Equal(nVectors, n);
            Assert.Equal(packed, p);
            Assert.Equal(scales, s);
            Assert.Equal(ids, slotToId);
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
        }
    }

    [Fact]
    public void TvimV1FileIsRejectedWithUpgradeHint()
    {
        string path = TestHelpers.TempPath(".tvim", "tvim-v1");
        try
        {
            using (var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None))
            using (var writer = new BinaryWriter(stream))
            {
                writer.Write("TVIM"u8.ToArray());
                writer.Write((byte)1);
                writer.Write((byte)4);
                writer.Write(32u);
                writer.Write(1u);
                writer.Write(new byte[(32 / 8) * 4]);
                writer.Write(1.0f);
                writer.Write(42UL);
            }

            Exception ex = Assert.ThrowsAny<Exception>(() => Io.LoadIdMap(path));
            Assert.Contains("turbovec ≤ 0.4.3", ex.Message);
            Assert.Contains("Rebuild", ex.Message);
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
        }
    }

    [Fact]
    public void TvGarbageFileRejectedWithoutUpgradeHint()
    {
        string path = TestHelpers.TempPath(".tv", "tv-garbage");
        try
        {
            using (var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None))
            using (var writer = new BinaryWriter(stream))
            {
                writer.Write("NOPE"u8.ToArray());
                writer.Write(new byte[32]);
            }

            Exception ex = Assert.ThrowsAny<Exception>(() => Io.Load(path));
            Assert.Contains("wrong magic", ex.Message);
            Assert.DoesNotContain("turbovec ≤ 0.4.3", ex.Message);
        }
        finally
        {
            TestHelpers.DeleteFileIfExists(path);
        }
    }
}
