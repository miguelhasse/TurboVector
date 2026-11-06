using System.Globalization;

internal static class ExampleOptions
{
    public static Dictionary<string, string> Parse(string[] args, params string[] allowed)
    {
        var allowedSet = new HashSet<string>(allowed, StringComparer.Ordinal);
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int i = 0; i < args.Length; i++)
        {
            string argument = args[i];
            if (!argument.StartsWith("--", StringComparison.Ordinal))
            {
                throw new ArgumentException($"Unexpected argument '{argument}'. Options must use --name value syntax.");
            }

            string name = argument[2..];
            if (!allowedSet.Contains(name))
            {
                throw new ArgumentException($"Unknown option '{argument}'.");
            }

            if (++i >= args.Length)
            {
                throw new ArgumentException($"Missing value for '{argument}'.");
            }

            if (!values.TryAdd(name, args[i]))
            {
                throw new ArgumentException($"Option '{argument}' was specified more than once.");
            }
        }

        return values;
    }

    public static int GetInt(
        IReadOnlyDictionary<string, string> values,
        string name,
        int defaultValue,
        int minimum = 1)
    {
        if (!values.TryGetValue(name, out string? text))
        {
            return defaultValue;
        }

        if (!int.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out int result)
            || result < minimum)
        {
            throw new ArgumentException($"--{name} must be an integer greater than or equal to {minimum}.");
        }

        return result;
    }

    public static ulong GetULong(
        IReadOnlyDictionary<string, string> values,
        string name,
        ulong defaultValue)
    {
        if (!values.TryGetValue(name, out string? text))
        {
            return defaultValue;
        }

        if (!ulong.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out ulong result))
        {
            throw new ArgumentException($"--{name} must be an unsigned integer.");
        }

        return result;
    }
}
