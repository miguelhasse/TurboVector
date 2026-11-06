using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace TurboVector.Benchmarks;

[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.HostProcess)]
public class SearchBenchmarks
{
    [Params(2, 4)]
    public int BitWidth { get; set; }

    [Params(1536)]
    public int Dim { get; set; }

    [Params(1_000, 10_000, 100_000)]
    public int N { get; set; }

    private TurboQuantIndex _index = null!;
    private float[] _queries = null!;
    private float[] _singleQuery = null!;

    [GlobalSetup]
    public void Setup()
    {
        _index = new TurboQuantIndex(Dim, BitWidth);
        _index.Add(BenchmarkData.MakeVectors(N, Dim));
        _queries = BenchmarkData.MakeVectors(100, Dim, seed: 4242);
        _singleQuery = _queries[..Dim];
    }

    [Benchmark]
    public float SearchSingleQuery()
    {
        var results = _index.Search(_singleQuery, 10);
        return results.Scores.Length == 0 ? 0f : results.Scores[0];
    }

    [Benchmark]
    public float SearchBatch()
    {
        float sum = 0f;
        for (int i = 0; i < 100; i++)
        {
            var results = _index.Search(_queries.AsSpan(i * Dim, Dim), 10);
            if (results.Scores.Length != 0)
            {
                sum += results.Scores[0];
            }
        }

        return sum;
    }
}
