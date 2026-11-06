using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace TurboVector.Benchmarks;

[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.HostProcess)]
public class IdMapSearchBenchmarks
{
    [Params(4)]
    public int BitWidth { get; set; }

    [Params(1536)]
    public int Dim { get; set; }

    [Params(10_000)]
    public int N { get; set; }

    [Params(0.1, 0.5)]
    public double AllowFraction { get; set; }

    private IdMapIndex _index = null!;
    private float[] _query = null!;
    private ulong[] _allowlist = null!;

    [GlobalSetup]
    public void Setup()
    {
        _index = new IdMapIndex(Dim, BitWidth);
        _index.AddWithIds(BenchmarkData.MakeVectors(N, Dim), Enumerable.Range(0, N).Select(static i => (ulong)i).ToArray());
        _query = BenchmarkData.MakeVectors(1, Dim, seed: 9001);
        _allowlist = BenchmarkData.MakeAllowlist(N, AllowFraction);
    }

    [Benchmark]
    public float IdMapSearchWithAllowlist()
    {
        var results = _index.SearchWithAllowlist(_query, 10, _allowlist);
        return results.Scores.Length == 0 ? 0f : results.Scores[0];
    }
}
