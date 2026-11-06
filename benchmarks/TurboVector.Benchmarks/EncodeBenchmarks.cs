using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace TurboVector.Benchmarks;

[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.HostProcess)]
public class EncodeBenchmarks
{
    [Params(2, 4)]
    public int BitWidth { get; set; }

    [Params(256, 1536)]
    public int Dim { get; set; }

    private float[] _vectors = null!;
    private float[] _rotation = null!;
    private float[] _boundaries = null!;
    private float[] _centroids = null!;

    [GlobalSetup]
    public void Setup()
    {
        _rotation = Rotation.MakeRotationMatrix(Dim);
        (_boundaries, _centroids) = Codebook.Compute(BitWidth, Dim);
        _vectors = BenchmarkData.MakeVectors(1000, Dim);
    }

    [Benchmark]
    public int Encode()
    {
        var (packedCodes, scales) = Encoder.Encode(_vectors, 1000, Dim, _rotation, _boundaries, _centroids, BitWidth);
        return packedCodes.Length + scales.Length;
    }
}
