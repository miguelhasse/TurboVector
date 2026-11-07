using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace TurboVector;

public static partial class Search
{
    private const uint PfArmV82I8mmInstructionsAvailable = 66;
    private const uint AtHwcap2 = 26;
    private const uint Hwcap2I8mm = 1u << 13;

    private static readonly Lazy<bool> Planes2Enabled = new(() =>
        Environment.GetEnvironmentVariable("TURBOVEC_2BIT_PLANES") != "0");

    private static readonly Lazy<bool> Planes4Enabled = new(() =>
        Environment.GetEnvironmentVariable("TURBOVEC_4BIT_PLANES") != "0");

    private static readonly Lazy<bool> Arm2BitVm8Enabled = new(() =>
        Environment.GetEnvironmentVariable("TURBOVEC_2BIT_VM8") == "1");

    private static readonly Lazy<bool> ArmI8mmSupported = new(() =>
    {
        if (Environment.GetEnvironmentVariable("TURBOVEC_NO_I8MM") is not null) return false;
        if (OperatingSystem.IsWindows()) return IsProcessorFeaturePresent(PfArmV82I8mmInstructionsAvailable);
        if (OperatingSystem.IsLinux() || OperatingSystem.IsAndroid()) return (GetAuxiliaryValue(AtHwcap2) & Hwcap2I8mm) != 0;
        if (OperatingSystem.IsMacOS() || OperatingSystem.IsIOS() || OperatingSystem.IsTvOS() || OperatingSystem.IsMacCatalyst())
        {
            var size = (nuint)sizeof(int);
            return SysctlByName("hw.optional.arm.FEAT_I8MM", out var enabled, ref size, 0, 0) == 0 && enabled != 0;
        }
        return false;
    });

    private static readonly Lazy<bool> VectorMajorEnabled = new(() =>
    {
        if (IsDisabled("TURBOVEC_NO_VECTOR_MAJOR") || IsDisabled("TURBOVEC_NO_VNNI"))
        {
            return false;
        }

        if (X86Base.IsSupported)
        {
            var features = X86Base.CpuId(7, 0);
            return Avx512F.IsSupported && Avx512BW.IsSupported && Avx512Vbmi.IsSupported
                && (features.Ecx & (1 << 11)) != 0;
        }

        return Dp.Arm64.IsSupported;
    });

    private static bool IsDisabled(string variable)
        => Environment.GetEnvironmentVariable(variable) is { } value && value != "0";

    private static bool UseVectorMajor(int bits, int nByteGroups)
        => nByteGroups % 4 == 0 && VectorMajorEnabled.Value && (X86Base.IsSupported || bits == 4);

    internal static bool UseArm2BitPermuteDot(int nByteGroups)
        => AdvSimd.Arm64.IsSupported && nByteGroups % 8 == 0
            && Arm2BitVm8Enabled.Value && VectorMajorEnabled.Value && ArmI8mmSupported.Value;

    internal static bool IsArmPermuteBatch(int query, int queryCount)
    {
        var width = queryCount >= 12 ? 12 : queryCount >= 8 ? 8 : 4;
        var size = Math.Min(width, queryCount - query / width * width);
        return size is 4 or 8 or 12;
    }

    internal static bool UsePlanes(int bits, int nByteGroups, int nVectors)
    {
        if (nVectors < 32_768) return false;
        if (AdvSimd.Arm64.IsSupported && Arm2BitVm8Enabled.Value) return false;
        if (bits == 2)
        {
            return nByteGroups % 8 == 0
                && Planes2Enabled.Value
                && (X86Base.IsSupported ? VectorMajorEnabled.Value : AdvSimd.Arm64.IsSupported);
        }

        return bits == 4 && nByteGroups % 16 == 0
            && Planes4Enabled.Value
            && VectorMajorEnabled.Value;
    }

    [LibraryImport("kernel32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool IsProcessorFeaturePresent(uint feature);

    [LibraryImport("libc", EntryPoint = "getauxval")]
    private static partial nuint GetAuxiliaryValue(nuint type);

    [LibraryImport("libSystem.B.dylib", EntryPoint = "sysctlbyname", StringMarshalling = StringMarshalling.Utf8)]
    private static partial int SysctlByName(string name, out int value, ref nuint size, nint newValue, nuint newSize);
}
