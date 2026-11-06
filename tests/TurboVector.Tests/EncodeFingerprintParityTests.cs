using System.Buffers.Binary;

namespace TurboVector.Tests;

public class EncodeFingerprintParityTests
{
    private static readonly (int Dim, int Bits, ulong Calibration, ulong Codes, ulong Scales)[] Goldens =
    [
        (200, 2, 0x1974331dac161d9b, 0x122b777da643eba6, 0x467e7fc11cbd63c8),
        (600, 3, 0x45a9253a5394ef8e, 0x2ade4149a0363781, 0x1f1a60f658173d8d),
        (768, 4, 0x675c5a5d5de10558, 0x2b6e66a18e412461, 0xe19782114e0d4434),
        (1000, 4, 0xe2a3800eca61b130, 0xe02d8df20dcb1dde, 0xd0700a74a602a059),
        (1024, 3, 0x570b4b3ef83a267d, 0x06a3982646ab3fb9, 0x647e88ac939e606e),
        (1536, 2, 0x5d9700e531f98c58, 0x65d6c36402627280, 0xd71c40bb564b1fcf),
        (1536, 4, 0x90eec90ef20017f3, 0xe06d3facabb1be0e, 0x9023d9253dbebb3a),
        (3072, 4, 0x8bb12d4a388dfa70, 0x2dbc750011989ee3, 0xc2155ee6e517cced),
    ];

    [Theory]
    [MemberData(nameof(GoldenCases))]
    public void EncodingMatchesPinnedRustFingerprints(
        int dim,
        int bits,
        ulong expectedCalibration,
        ulong expectedCodes,
        ulong expectedScales)
    {
        const int rows = 2000;
        var vectors = CreateVectors(rows, dim, CellSeed(dim, bits));
        var index = new TurboQuantIndex(dim, bits);

        index.Calibrate(vectors);
        index.Add(vectors);

        Assert.Equal(CalibrationState.Calibrated, index.CalibrationState);
        Assert.Equal(expectedCalibration, HashCalibration(index.TqPlusShift, index.TqPlusScale));
        Assert.Equal(expectedCodes, Hash(index.PackedCodes));
        Assert.Equal(expectedScales, Hash(index.Scales));
    }

    public static IEnumerable<object[]> GoldenCases()
    {
        foreach (var item in Goldens)
        {
            yield return [item.Dim, item.Bits, item.Calibration, item.Codes, item.Scales];
        }
    }

    private static float[] CreateVectors(int rows, int dim, ulong seed)
    {
        float[] vectors = new float[checked(rows * dim)];
        ulong state = seed | 1UL;
        for (int i = 0; i < vectors.Length; i++)
        {
            state = unchecked(state * 6_364_136_223_846_793_005UL + 1_442_695_040_888_963_407UL);
            uint high = (uint)(state >> 32);
            vectors[i] = (float)(high / 2_147_483_648.0 - 1.0);
        }

        return vectors;
    }

    private static ulong CellSeed(int dim, int bits)
        => 0x7B0_0000UL ^ ((ulong)dim << 8) ^ (uint)bits;

    private static ulong HashCalibration(ReadOnlySpan<float> shift, ReadOnlySpan<float> scale)
    {
        float[] values = new float[shift.Length + scale.Length];
        shift.CopyTo(values);
        scale.CopyTo(values.AsSpan(shift.Length));
        return Hash(values);
    }

    private static ulong Hash(ReadOnlySpan<float> values)
    {
        ulong hash = 0xcbf29ce484222325;
        Span<byte> bytes = stackalloc byte[4];
        foreach (float value in values)
        {
            BinaryPrimitives.WriteInt32LittleEndian(bytes, BitConverter.SingleToInt32Bits(value));
            hash = AddByte(hash, bytes[0]);
            hash = AddByte(hash, bytes[1]);
            hash = AddByte(hash, bytes[2]);
            hash = AddByte(hash, bytes[3]);
        }

        return hash;
    }

    private static ulong Hash(ReadOnlySpan<byte> values)
    {
        ulong hash = 0xcbf29ce484222325;
        foreach (byte value in values)
        {
            hash = AddByte(hash, value);
        }

        return hash;
    }

    private static ulong AddByte(ulong hash, byte value)
        => unchecked((hash ^ value) * 0x100000001b3);
}
