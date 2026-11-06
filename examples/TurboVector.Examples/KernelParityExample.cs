using TurboVector;

internal static class KernelParityExample
{
    public static void Run(string[] args)
    {
        var options = ExampleOptions.Parse(args, "dim", "bits", "seed", "n-db", "n-queries", "k");
        int dim = ExampleOptions.GetInt(options, "dim", 1536);
        int bits = ExampleOptions.GetInt(options, "bits", 4);
        ulong seed = ExampleOptions.GetULong(options, "seed", 42);
        int databaseCount = ExampleOptions.GetInt(options, "n-db", 5000);
        int queryCount = ExampleOptions.GetInt(options, "n-queries", 50);
        int k = ExampleOptions.GetInt(options, "k", 32);
        if (k > databaseCount)
        {
            throw new ArgumentException("--k cannot exceed --n-db.");
        }

        float[] database = new float[checked(databaseCount * dim)];
        float[] queries = new float[checked(queryCount * dim)];
        ExampleData.FillNormal(database, seed);
        ExampleData.FillNormal(queries, unchecked(seed + 1));
        ExampleData.NormalizeRows(database, dim);
        ExampleData.NormalizeRows(queries, dim);

        Console.Error.WriteLine(
            $"# arch={System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture} " +
            $"dim={dim} bits={bits} seed={seed} n_db={databaseCount} n_queries={queryCount} k={k}");

        var index = new TurboQuantIndex(dim, bits);
        index.Add(database);
        index.Prepare();
        SearchResults results = index.Search(queries, k);

        for (int qi = 0; qi < queryCount; qi++)
        {
            long[] indices = results.IndicesForQuery(qi).ToArray();
            Array.Sort(indices);
            foreach (long vectorIndex in indices)
            {
                Console.WriteLine($"{qi}\t{vectorIndex}");
            }
        }
    }
}
