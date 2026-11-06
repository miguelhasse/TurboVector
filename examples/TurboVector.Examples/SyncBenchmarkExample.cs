using TurboVector;

internal static class SyncBenchmarkExample
{
    public static void Run(string[] args)
    {
        var options = ExampleOptions.Parse(args, "dim", "bits", "n", "append-repeats", "removes", "scattered");
        int dim = ExampleOptions.GetInt(options, "dim", 128);
        int bits = ExampleOptions.GetInt(options, "bits", 2);
        int count = ExampleOptions.GetInt(options, "n", 10_000);
        int appendRepeats = ExampleOptions.GetInt(options, "append-repeats", 3, minimum: 0);
        int removes = ExampleOptions.GetInt(options, "removes", 5, minimum: 0);
        int scattered = ExampleOptions.GetInt(options, "scattered", 20, minimum: 0);

        string directory = Path.Combine(Path.GetTempPath(), $"turbovec-sync-bench-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        string path = Path.Combine(directory, "bench.tv");
        try
        {
            float[] initial = ExampleData.CreateLcgVectors(count, dim, 0x243F_6A88_85A3_08D3);
            var index = new TurboQuantIndex(dim, bits);
            index.Add(initial);
            var timer = System.Diagnostics.Stopwatch.StartNew();
            index.Sync(path);
            timer.Stop();
            Console.WriteLine($"first sync (full write): {timer.Elapsed.TotalMilliseconds:F2} ms");

            double bestAppend = double.PositiveInfinity;
            for (int i = 0; i < appendRepeats; i++)
            {
                float[] added = ExampleData.CreateLcgVectors(32, dim, 0xA55E3UL + (ulong)i);
                index.Add(added);
                timer.Restart();
                index.Sync(path);
                timer.Stop();
                bestAppend = Math.Min(bestAppend, timer.Elapsed.TotalMilliseconds);
            }

            if (appendRepeats > 0)
            {
                Console.WriteLine($"append 32 rows: best {bestAppend:F2} ms");
            }

            int removalCount = Math.Min(removes, index.Len);
            double bestRemoval = 0;
            for (int i = 0; i < removalCount; i++)
            {
                index.SwapRemove(Math.Min(100 + i, index.Len - 1));
                timer.Restart();
                index.Sync(path);
                timer.Stop();
                bestRemoval = i == 0 ? timer.Elapsed.TotalMilliseconds : Math.Min(bestRemoval, timer.Elapsed.TotalMilliseconds);
            }

            if (removalCount > 0)
            {
                Console.WriteLine($"single removal: best {bestRemoval:F2} ms");
            }

            int scatteredCount = Math.Min(scattered, Math.Max(0, index.Len / 4));
            if (scatteredCount > 0)
            {
                int maxVictim = Math.Max(0, index.Len - scatteredCount - 1);
                int step = Math.Max(1, maxVictim / scatteredCount);
                int[] victims = Enumerable.Range(0, scatteredCount)
                    .Select(i => Math.Min(maxVictim, 7 + i * step))
                    .Distinct()
                    .OrderByDescending(static i => i)
                    .ToArray();
                foreach (int victim in victims)
                {
                    index.SwapRemove(victim);
                }

                timer.Restart();
                index.Sync(path);
                timer.Stop();
                Console.WriteLine($"{victims.Length} scattered removals, one sync: {timer.Elapsed.TotalMilliseconds:F2} ms");
            }
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
