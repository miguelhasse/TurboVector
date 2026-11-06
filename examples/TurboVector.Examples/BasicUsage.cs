using TurboVector;

internal static class BasicUsage
{
    private const ulong Multiplier = 6364136223846793005UL;
    private const ulong Increment = 1442695040888963407UL;

    public static void Run()
    {
        const int vectorCount = 100;
        const int dim = 64;
        const int bitWidth = 4;

        float[] data = CreateNormalizedDataset(vectorCount, dim, 0xBADC0FFEE0DDF00DUL);
        float[] query = data.AsSpan(0, dim).ToArray();

        var index = new TurboQuantIndex(dim, bitWidth);
        index.Add(data);
        Console.WriteLine($"Indexed {index.Len} vectors (dim={index.Dim}, bits={index.BitWidth}).");

        SearchResults results = index.Search(query, 5);
        if (results.Indices.Length == 0)
        {
            throw new InvalidOperationException("Search returned no results.");
        }

        Console.WriteLine($"Top result for vector 0: slot={results.Indices[0]}, score={results.Scores[0]:F4}");
        if (results.Indices[0] != 0)
        {
            throw new InvalidOperationException($"Expected slot 0 as top result, got {results.Indices[0]}.");
        }

        PrintTopResults(results);

        string path = Path.Combine(Path.GetTempPath(), $"turbovec-basic-{Guid.NewGuid():N}.tv");
        try
        {
            index.Write(path);
            TurboQuantIndex reloaded = TurboQuantIndex.Load(path);
            SearchResults reloadedResults = reloaded.Search(query, 5);
            if (reloadedResults.Indices.Length == 0 || reloadedResults.Indices[0] != results.Indices[0])
            {
                throw new InvalidOperationException("Reloaded index changed the top result.");
            }

            Console.WriteLine($"Reloaded index top result: slot={reloadedResults.Indices[0]}, score={reloadedResults.Scores[0]:F4}");
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    private static void PrintTopResults(SearchResults results)
    {
        Console.WriteLine("Top-5 results:");
        for (int i = 0; i < results.K; i++)
        {
            Console.WriteLine($"  {i + 1}. slot={results.Indices[i]}, score={results.Scores[i]:F4}");
        }
    }

    private static float[] CreateNormalizedDataset(int vectorCount, int dim, ulong seed)
    {
        float[] data = new float[vectorCount * dim];
        ulong state = seed;

        for (int i = 0; i < data.Length; i++)
        {
            state = unchecked((state * Multiplier) + Increment);
            uint bits = ((uint)(state >> 32) & 0x007F_FFFFu) | 0x3F80_0000u;
            float value = ((BitConverter.UInt32BitsToSingle(bits) - 1.0f) * 2.0f) - 1.0f;
            data[i] = value;
        }

        for (int offset = 0; offset < data.Length; offset += dim)
        {
            Normalize(data.AsSpan(offset, dim));
        }

        return data;
    }

    private static void Normalize(Span<float> vector)
    {
        double sum = 0.0;
        for (int i = 0; i < vector.Length; i++)
        {
            sum += vector[i] * vector[i];
        }

        float scale = (float)(1.0 / Math.Sqrt(sum));
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] *= scale;
        }
    }
}
