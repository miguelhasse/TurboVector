using System.Buffers.Binary;
using TurboVector;

namespace TurboVector.Tests;

public class CodebookParityTests
{
    private static readonly (int Dim, int Bits, ulong Boundaries, ulong Centroids)[] Goldens =
    [
        (200, 2, 0x4fb0378dc1d86d75, 0xd43ff8712da19915),
        (768, 4, 0x27273d875c560161, 0x09704dad5b65a00d),
        (1024, 3, 0xd9eeefae66c34fd1, 0xfcf5342e5aefb105),
        (1536, 2, 0xb1a97993603c7dcd, 0x1348c9ae60bbdc3d),
        (1536, 4, 0x05f7a92f87aba9a1, 0x84f2c25cecbf6761),
        (3072, 4, 0x9a1b3dca8251faad, 0x6a56a1738e12e62d),
    ];

    [Theory]
    [MemberData(nameof(GoldenCases))]
    public void CodebookMatchesPinnedRustHashes(int dim, int bits, ulong expectedBoundaries, ulong expectedCentroids)
    {
        var (boundaries, centroids) = Codebook.Compute(bits, dim);
        Assert.Equal(expectedBoundaries, Hash(boundaries));
        Assert.Equal(expectedCentroids, Hash(centroids));
    }

    public static IEnumerable<object[]> GoldenCases()
    {
        foreach (var item in Goldens)
        {
            yield return [item.Dim, item.Bits, item.Boundaries, item.Centroids];
        }
    }

    private static ulong Hash(ReadOnlySpan<float> values)
    {
        ulong hash = 0xcbf29ce484222325;
        Span<byte> bytes = stackalloc byte[4];
        foreach (float value in values)
        {
            BinaryPrimitives.WriteInt32LittleEndian(bytes, BitConverter.SingleToInt32Bits(value));
            for (int i = 0; i < bytes.Length; i++)
            {
                hash ^= bytes[i];
                hash *= 0x100000001b3;
            }
        }

        return hash;
    }
}
