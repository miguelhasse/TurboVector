namespace TurboVector;

internal static class InputValidation
{
    internal static void ValidateBitWidth(int bitWidth, string parameterName)
    {
        if (bitWidth is not (2 or 3 or 4))
        {
            throw new ArgumentOutOfRangeException(parameterName, "bitWidth must be 2, 3, or 4.");
        }
    }

    internal static void ValidateDimension(int dim, string parameterName)
    {
        if (dim <= 0 || dim > Constants.MaxDim || dim % 8 != 0)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                $"dim must be positive, divisible by 8, and no greater than {Constants.MaxDim}.");
        }
    }

    internal static void ValidateValues(ReadOnlySpan<float> values, int dim, string parameterName)
    {
        for (int i = 0; i < values.Length; i++)
        {
            float value = values[i];
            if (!float.IsFinite(value) || MathF.Abs(value) >= Constants.MaxInputMagnitude)
            {
                throw new ArgumentException(
                    $"{parameterName} value at row {i / dim}, coordinate {i % dim} must be finite and have magnitude less than {Constants.MaxInputMagnitude}.",
                    parameterName);
            }
        }
    }
}
