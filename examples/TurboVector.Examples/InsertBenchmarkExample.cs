using System.Text.Json;
using System.Text.Json.Serialization;
using TurboVector;

internal static class InsertBenchmarkExample
{
    public static void Run(string[] args)
    {
        var options = ExampleOptions.Parse(args, "dim", "bits", "n", "append", "singles", "removes", "repeats");
        int dim = ExampleOptions.GetInt(options, "dim", 128);
        int bits = ExampleOptions.GetInt(options, "bits", 2);
        int count = ExampleOptions.GetInt(options, "n", 5000, minimum: 2);
        int appendCount = ExampleOptions.GetInt(options, "append", 256, minimum: 0);
        int singlesCount = ExampleOptions.GetInt(options, "singles", 100, minimum: 0);
        int removeCount = ExampleOptions.GetInt(options, "removes", 500, minimum: 0);
        int repeats = ExampleOptions.GetInt(options, "repeats", 3);

        float[] database = ExampleData.CreateNormalizedUniformVectors(count, dim, 0xD1536);
        float[] added = ExampleData.CreateNormalizedUniformVectors(
            checked(appendCount + singlesCount), dim, 0xA55E3);
        ReadOnlySpan<float> batch = added.AsSpan(0, checked(appendCount * dim));
        ReadOnlySpan<float> singles = added.AsSpan(checked(appendCount * dim));
        var bulk = new double[repeats];
        var warm = new double[repeats];
        var single = new double[repeats];
        var idMapRemove = new double[repeats];
        var swapRemove = new double[repeats];

        for (int run = 0; run < repeats; run++)
        {
            var index = new TurboQuantIndex(dim, bits);
            var timer = System.Diagnostics.Stopwatch.StartNew();
            index.Calibrate(database);
            index.Add(database);
            timer.Stop();
            bulk[run] = count / timer.Elapsed.TotalSeconds;

            timer.Restart();
            index.Add(batch);
            timer.Stop();
            warm[run] = appendCount == 0 ? 0 : appendCount / timer.Elapsed.TotalSeconds;
        }

        for (int run = 0; run < repeats; run++)
        {
            var index = new TurboQuantIndex(dim, bits);
            index.Calibrate(database);
            index.Add(database);
            var timer = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < singlesCount; i++)
            {
                index.Add(singles.Slice(i * dim, dim));
            }

            timer.Stop();
            single[run] = singlesCount == 0 ? 0 : timer.Elapsed.TotalMilliseconds * 1000 / singlesCount;
        }

        int removes = Math.Min(removeCount, count);
        if (removes > 0)
        {
            ulong[] ids = new ulong[count];
            for (int i = 0; i < ids.Length; i++)
            {
                ids[i] = (ulong)i;
            }

            for (int run = 0; run < repeats; run++)
            {
                var idMap = new IdMapIndex(dim, bits);
                idMap.AddWithIds(database, ids);
                var timer = System.Diagnostics.Stopwatch.StartNew();
                for (int id = 0; id < removes; id++)
                {
                    idMap.Remove((ulong)id);
                }

                timer.Stop();
                idMapRemove[run] = timer.Elapsed.TotalMilliseconds * 1000 / removes;

                var index = new TurboQuantIndex(dim, bits);
                index.Add(database);
                timer.Restart();
                for (int i = 0; i < removes; i++)
                {
                    index.SwapRemove(0);
                }

                timer.Stop();
                swapRemove[run] = timer.Elapsed.TotalMilliseconds * 1000 / removes;
            }
        }

        var result = new
        {
            dim,
            bit_width = bits,
            repeats,
            bulk_insert_vecs_per_sec = Median(bulk),
            warm_insert_vecs_per_sec = Median(warm),
            single_add_us = Median(single),
            idmap_remove_us = removes == 0 ? 0 : Median(idMapRemove),
            swap_remove_us = removes == 0 ? 0 : Median(swapRemove),
        };
        Console.WriteLine(JsonSerializer.Serialize(result, new JsonSerializerOptions
        {
            WriteIndented = true,
            NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
        }));
    }

    private static double Median(double[] values)
    {
        Array.Sort(values);
        return values[values.Length / 2];
    }
}
