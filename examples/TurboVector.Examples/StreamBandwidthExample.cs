internal static class StreamBandwidthExample
{
    public static void Run(string[] args)
    {
        var options = ExampleOptions.Parse(args, "small-mib", "large-mib", "threads");
        int smallMiB = ExampleOptions.GetInt(options, "small-mib", 32);
        int largeMiB = ExampleOptions.GetInt(options, "large-mib", 128);
        int threads = ExampleOptions.GetInt(options, "threads", Environment.ProcessorCount);
        if (largeMiB < smallMiB)
        {
            throw new ArgumentException("--large-mib must be greater than or equal to --small-mib.");
        }

        double small = Measure(smallMiB, threads, repetitions: 4);
        double large = Measure(largeMiB, threads, repetitions: 2);
        double single = Measure(largeMiB, threads: 1, repetitions: 1);

        Console.WriteLine($"threads {threads}");
        Console.WriteLine();
        Console.WriteLine("read bandwidth");
        Console.WriteLine($"  {smallMiB,4} MiB (cache-sized sample) {small,8:F1} GB/s");
        Console.WriteLine($"  {largeMiB,4} MiB (larger sample)     {large,8:F1} GB/s");
        Console.WriteLine($"  {largeMiB,4} MiB, single thread      {single,8:F1} GB/s");
        Console.WriteLine();
        Console.WriteLine("A much higher smaller-buffer result suggests more cache reuse.");
        Console.WriteLine("Compare these figures with bytes scanned per search to estimate bandwidth headroom.");
    }

    private static double Measure(int mebibytes, int threads, int repetitions)
    {
        int elementCount = checked(mebibytes * 1024 * 1024 / sizeof(ulong));
        ulong[] buffer = GC.AllocateUninitializedArray<ulong>(elementCount);
        Array.Fill(buffer, 1UL);

        ulong warm = 0;
        for (int i = 0; i < buffer.Length; i += 512)
        {
            warm += buffer[i];
        }

        GC.KeepAlive(warm);
        var partials = new ulong[threads];
        var options = new ParallelOptions { MaxDegreeOfParallelism = threads };
        double bestSeconds = double.PositiveInfinity;
        for (int repetition = 0; repetition < repetitions; repetition++)
        {
            Array.Clear(partials);
            var timer = System.Diagnostics.Stopwatch.StartNew();
            Parallel.For(0, threads, options, worker =>
            {
                int start = (int)((long)buffer.Length * worker / threads);
                int end = (int)((long)buffer.Length * (worker + 1) / threads);
                ulong sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
                ulong sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0;
                int i = start;
                for (; i + 7 < end; i += 8)
                {
                    sum0 += buffer[i];
                    sum1 += buffer[i + 1];
                    sum2 += buffer[i + 2];
                    sum3 += buffer[i + 3];
                    sum4 += buffer[i + 4];
                    sum5 += buffer[i + 5];
                    sum6 += buffer[i + 6];
                    sum7 += buffer[i + 7];
                }

                ulong sum = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;
                for (; i < end; i++)
                {
                    sum += buffer[i];
                }

                partials[worker] = sum;
            });

            timer.Stop();
            bestSeconds = Math.Min(bestSeconds, timer.Elapsed.TotalSeconds);
        }

        ulong sink = 0;
        foreach (ulong partial in partials)
        {
            sink += partial;
        }

        GC.KeepAlive(sink);
        long bytes = checked((long)mebibytes * 1024 * 1024);
        return (double)bytes / bestSeconds / 1e9;
    }
}
