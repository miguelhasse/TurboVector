using System.Buffers.Binary;
using TurboVector;

internal static class DumpState
{
    public static void Run(string outputDirectory = "state_output")
    {
        Directory.CreateDirectory(outputDirectory);

        foreach ((int dim, int bits) in new[] { (200, 2), (200, 4), (1536, 2), (1536, 4) })
        {
            Console.WriteLine($"Computing state for dim={dim}, bits={bits}...");
            float[] rotation = Rotation.MakeRotationMatrix(dim);
            var (boundaries, centroids) = Codebook.Compute(bits, dim);

            string path = Path.Combine(outputDirectory, $"state_d{dim}_b{bits}.bin");
            using var stream = File.Create(path);
            using var writer = new BinaryWriter(stream);
            WriteFloatArray(writer, rotation);
            WriteFloatArray(writer, boundaries);
            WriteFloatArray(writer, centroids);

            Console.WriteLine($"Wrote {path}");
        }
    }

    private static void WriteFloatArray(BinaryWriter writer, ReadOnlySpan<float> values)
    {
        Span<byte> buffer = stackalloc byte[sizeof(float)];
        foreach (float value in values)
        {
            BinaryPrimitives.WriteSingleLittleEndian(buffer, value);
            writer.Write(buffer);
        }
    }
}
