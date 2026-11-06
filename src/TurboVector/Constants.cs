namespace TurboVector;

internal static class Constants
{
    internal const int Block = 32;
    internal const int FlushEvery = 256;
    internal const int MaxDim = 16_384;
    internal const float MaxInputMagnitude = 1e16f;
    internal const ulong RotationSeed = 42UL;
}
