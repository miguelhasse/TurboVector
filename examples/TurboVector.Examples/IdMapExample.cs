using TurboVector;

internal static class IdMapExample
{
    private const ulong Multiplier = 6364136223846793005UL;
    private const ulong Increment = 1442695040888963407UL;

    public static void Run()
    {
        const int vectorCount = 50;
        const int dim = 32;
        const int bitWidth = 4;

        float[] data = CreateNormalizedDataset(vectorCount, dim, 0x1234_5678_9ABC_DEF0UL);
        ulong[] ids = Enumerable.Range(0, vectorCount).Select(i => 1001UL + (ulong)i).ToArray();
        float[] query = data.AsSpan(0, dim).ToArray();

        var index = new IdMapIndex(dim, bitWidth);
        index.AddWithIds(data, ids);
        Console.WriteLine($"Indexed {index.Len} vectors with IDs {ids[0]}..{ids[^1]}.");

        var (scores, foundIds) = index.Search(query, 5);
        if (foundIds.Length == 0)
        {
            throw new InvalidOperationException("Search returned no results.");
        }

        Console.WriteLine($"Top result before removal: id={foundIds[0]}, score={scores[0]:F4}");
        if (foundIds[0] != 1001UL)
        {
            throw new InvalidOperationException($"Expected ID 1001 as top result, got {foundIds[0]}.");
        }

        bool removed = index.Remove(1001UL);
        if (!removed)
        {
            throw new InvalidOperationException("Expected to remove ID 1001.");
        }

        var (scoresAfterRemove, idsAfterRemove) = index.Search(query, 5);
        if (idsAfterRemove.Contains(1001UL))
        {
            throw new InvalidOperationException("Removed ID 1001 still appears in search results.");
        }

        Console.WriteLine($"Top result after removal: id={idsAfterRemove[0]}, score={scoresAfterRemove[0]:F4}");

        string path = Path.Combine(Path.GetTempPath(), $"turbovec-idmap-{Guid.NewGuid():N}.tvim");
        try
        {
            index.Write(path);
            IdMapIndex reloaded = IdMapIndex.Load(path);
            var (_, reloadedIds) = reloaded.Search(query, 5);
            if (reloadedIds.Contains(1001UL))
            {
                throw new InvalidOperationException("Reloaded index unexpectedly contains removed ID 1001 in results.");
            }

            Console.WriteLine($"Reloaded index top result: id={reloadedIds[0]}");
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
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
