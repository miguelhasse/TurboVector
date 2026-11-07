namespace TurboVector;

internal static class Pack
{
    internal static void ExtractCodes(ReadOnlySpan<byte> packedCodes, int nVectors, int bits, int dim, Span<byte> codes)
    {
        int bytesPerPlane = dim / 8;
        int bytesPerRow = checked(bytesPerPlane * bits);
        int expectedLength = checked(bytesPerRow * nVectors);
        if (packedCodes.Length != expectedLength)
        {
            throw new ArgumentException($"Expected packedCodes length {expectedLength}, got {packedCodes.Length}.", nameof(packedCodes));
        }

        int expectedCodeLength = checked(nVectors * dim);
        if (codes.Length != expectedCodeLength)
        {
            throw new ArgumentException($"Expected codes length {expectedCodeLength}, got {codes.Length}.", nameof(codes));
        }

        for (int row = 0; row < nVectors; row++)
        {
            int packedRowStart = row * bytesPerRow;
            int codeRowStart = row * dim;
            for (int d = 0; d < dim; d++)
            {
                int byteInPlane = d >> 3;
                int bit = 7 - (d & 7);
                byte code = 0;
                for (int plane = 0; plane < bits; plane++)
                {
                    if ((packedCodes[packedRowStart + plane * bytesPerPlane + byteInPlane] & (1 << bit)) != 0)
                    {
                        code |= (byte)(1 << plane);
                    }
                }

                codes[codeRowStart + d] = code;
            }
        }
    }

    internal static (byte[] Blocked, int BlockCount) Repack(ReadOnlySpan<byte> packedCodes, int nVectors, int bits, int dim)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(nVectors);

        if (bits <= 0 || bits > 8)
        {
            throw new ArgumentOutOfRangeException(nameof(bits));
        }

        if (dim < 0 || dim % 8 != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dim), "dim must be non-negative and divisible by 8.");
        }

        var bytesPerPlane = dim / 8;
        var codesPerByte = 8 / bits;
        if (codesPerByte == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(bits));
        }

        var nByteGroups = dim / codesPerByte;
        var bitsPerCode = bits == 3 ? 4 : bits;
        var nBlocks = (nVectors + Constants.Block - 1) / Constants.Block;
        var blockedSize = checked(nBlocks * nByteGroups * Constants.Block);
        var bytesPerRow = checked(bits * bytesPerPlane);
        var expectedPackedLength = checked(bytesPerRow * nVectors);
        if (packedCodes.Length != expectedPackedLength)
        {
            throw new ArgumentException($"Expected packedCodes length {expectedPackedLength}, got {packedCodes.Length}.", nameof(packedCodes));
        }

        // Write directly into the blocked output without the intermediate jagged array.
        var blocked = new byte[blockedSize];
        for (var blockIdx = 0; blockIdx < nBlocks; blockIdx++)
        {
            var baseVec = blockIdx * Constants.Block;
            for (var g = 0; g < nByteGroups; g++)
            {
                var outOffset = (blockIdx * nByteGroups + g) * Constants.Block;
                var dimStart = g * codesPerByte;
                for (var lane = 0; lane < Constants.Block; lane++)
                {
                    var vecIdx = baseVec + lane;
                    if (vecIdx >= nVectors)
                    {
                        break;
                    }

                    var vecRowStart = vecIdx * bytesPerRow;
                    byte byteVal = 0;
                    for (var c = 0; c < codesPerByte; c++)
                    {
                        var j = dimStart + c;
                        var byteInPlane = j / 8;
                        var bitInByte = 7 - (j % 8);
                        var bitMask = (byte)(1 << bitInByte);

                        byte code = 0;
                        for (var p = 0; p < bits; p++)
                        {
                            var planeByte = packedCodes[vecRowStart + p * bytesPerPlane + byteInPlane];
                            if ((planeByte & bitMask) != 0)
                            {
                                code |= (byte)(1 << p);
                            }
                        }

                        byteVal |= (byte)(code << ((codesPerByte - 1 - c) * bitsPerCode));
                    }

                    blocked[outOffset + lane] = byteVal;
                }
            }
        }

        return (blocked, nBlocks);
    }

    internal static (byte[] SubCodes, byte[] Plane2Blocked, int BlockCount) Repack3Bit(ReadOnlySpan<byte> packedCodes, int nVectors, int dim)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(nVectors);

        if (dim < 0 || dim % 8 != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dim), "dim must be non-negative and divisible by 8.");
        }

        var bytesPerPlane = dim / 8;
        var bytesPerRow = checked(3 * bytesPerPlane);
        var expectedPackedLength = checked(bytesPerRow * nVectors);
        if (packedCodes.Length != expectedPackedLength)
        {
            throw new ArgumentException($"Expected packedCodes length {expectedPackedLength}, got {packedCodes.Length}.", nameof(packedCodes));
        }

        var nBlocks = (nVectors + Constants.Block - 1) / Constants.Block;
        var subByteGroups = dim / 4;
        var subCodes = new byte[checked(nBlocks * subByteGroups * Constants.Block)];

        var plane2ByteGroups = bytesPerPlane;
        var plane2Blocked = new byte[checked(nBlocks * plane2ByteGroups * Constants.Block)];

        for (var blockIdx = 0; blockIdx < nBlocks; blockIdx++)
        {
            var baseVec = blockIdx * Constants.Block;

            for (var g = 0; g < subByteGroups; g++)
            {
                var outOffset = (blockIdx * subByteGroups + g) * Constants.Block;
                for (var lane = 0; lane < Constants.Block; lane++)
                {
                    var vecIdx = baseVec + lane;
                    if (vecIdx >= nVectors)
                    {
                        break;
                    }

                    var vecRowStart = vecIdx * bytesPerRow;
                    byte byteVal = 0;
                    var dimStart = g * 4;
                    var byteInPlane = g >> 1;
                    var planeByte0 = packedCodes[vecRowStart + byteInPlane];
                    var planeByte1 = packedCodes[vecRowStart + bytesPerPlane + byteInPlane];
                    for (var c = 0; c < 4; c++)
                    {
                        var j = dimStart + c;
                        var bitInByte = 7 - (j & 7);
                        var mask = (byte)(1 << bitInByte);
                        byte code = (byte)(
                            ((planeByte0 & mask) != 0 ? 1 : 0)
                            | ((planeByte1 & mask) != 0 ? 2 : 0));

                        byteVal |= (byte)(code << ((3 - c) * 2));
                    }

                    subCodes[outOffset + lane] = byteVal;
                }
            }

            for (var g = 0; g < plane2ByteGroups; g++)
            {
                var outOffset = (blockIdx * plane2ByteGroups + g) * Constants.Block;
                for (var lane = 0; lane < Constants.Block; lane++)
                {
                    var vecIdx = baseVec + lane;
                    if (vecIdx >= nVectors)
                    {
                        break;
                    }

                    var vecRowStart = vecIdx * bytesPerRow;
                    plane2Blocked[outOffset + lane] = packedCodes[vecRowStart + 2 * bytesPerPlane + g];
                }
            }
        }

        return (subCodes, plane2Blocked, nBlocks);
    }

    internal static void Expand3BitBlocked(
        ReadOnlySpan<byte> subCodes,
        ReadOnlySpan<byte> plane2Blocked,
        int dim,
        int nBlocks,
        Span<byte> destination)
    {
        var groups = dim / 2;
        var subGroups = dim / 4;
        var planeGroups = dim / 8;
        for (var block = 0; block < nBlocks; block++)
        {
            for (var group = 0; group < groups; group++)
            {
                var subOffset = (block * subGroups + group / 2) * Constants.Block;
                var planeOffset = (block * planeGroups + group / 4) * Constants.Block;
                var outputOffset = (block * groups + group) * Constants.Block;
                var subShift = (1 - (group & 1)) * 4;
                var planeShift = 6 - (group & 3) * 2;
                for (var lane = 0; lane < Constants.Block; lane++)
                {
                    var low = (subCodes[subOffset + lane] >> subShift) & 15;
                    var high = (plane2Blocked[planeOffset + lane] >> planeShift) & 3;
                    destination[outputOffset + lane] = (byte)(((low & 12) << 2) | ((high & 2) << 5)
                        | (low & 3) | ((high & 1) << 2));
                }
            }
        }
    }
}
