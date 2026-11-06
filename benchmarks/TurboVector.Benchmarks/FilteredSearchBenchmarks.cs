using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace TurboVector.Benchmarks;

[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.HostProcess)]
public class FilteredSearchBenchmarks
{
    [Params(4)]
    public int BitWidth { get; set; }

    [Params(1536)]
    public int Dim { get; set; }

    [Params(10_000)]
    public int N { get; set; }

    [Params(0.1, 0.5, 1.0)]
    public double AllowFraction { get; set; }

    private TurboQuantIndex _index = null!;
    private float[] _query = null!;
    private bool[] _mask = null!;

    [GlobalSetup]
    public void Setup()
    {
        _index = new TurboQuantIndex(Dim, BitWidth);
        _index.Add(BenchmarkData.MakeVectors(N, Dim));
        _query = BenchmarkData.MakeVectors(1, Dim, seed: 2024);
        _mask = BenchmarkData.MakeMask(N, AllowFraction);
    }

    [Benchmark]
    public float SearchWithMask()
    {
        var results = _index.SearchWithMask(_query, 10, _mask);
        return results.Scores.Length == 0 ? 0f : results.Scores[0];
    }
}
