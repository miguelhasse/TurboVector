# TurboVector

A C# .NET port of [turbovec](https://github.com/RyanCodrai/turbovec) — a **RaBitQ / TurboQuant** vector quantization library for high-dimensional approximate nearest-neighbour (ANN) search. Vectors are compressed to 2–4 bits per coordinate with near-optimal distortion, then scored using SIMD-accelerated look-up tables.

## What is TurboQuant?

TurboQuant is a **data-oblivious** scalar quantizer. Unlike traditional ANN indexes (IVF, HNSW, PQ) it requires no training, no clustering, and no warm-up data. The quantization grid is derived analytically from the distribution of coordinates on a rotated unit sphere (a Beta distribution), and an orthogonal rotation is pre-computed deterministically from a seed. This makes it ideal for:

- Streaming/online indexing where data arrives incrementally
- Situations where training data is unavailable or expensive
- Applications requiring a fully deterministic, reproducible index

Inner-product scores are corrected with a per-vector scale factor so that the self-score of every vector is ≈ 1.0.

## Solution Structure

```
TurboVector.slnx
├── src/TurboVector/          # Class library — the core algorithm
├── tests/TurboVector.Tests/  # xUnit.v3 unit & integration tests (84 tests)
├── examples/TurboVector.Examples/  # Console app showing common usage patterns
└── benchmarks/TurboVector.Benchmarks/  # BenchmarkDotNet performance measurements
```

## Requirements

- [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)

## Quick Start

```bash
# Build everything
dotnet build TurboVector.slnx

# Run tests
dotnet test tests/TurboVector.Tests

# Run examples
dotnet run --project examples/TurboVector.Examples

# Run benchmarks (Release mode required)
dotnet run --project benchmarks/TurboVector.Benchmarks -c Release -- --filter "*Search*"
```

## Usage at a Glance

Dimensions must be positive multiples of 8 up to 16,384, and `bitWidth` must be 2, 3, or 4. Vector and query coordinates must be finite and have magnitude below `1e16`. `.tv` and `.tvim` snapshots use the Rust-compatible v8 container, with v7/v8 loading, TQ+ calibration persistence, stream APIs, and atomic replacement. `Sync(path)` durably appends complete units where possible, uses atomic replacement when changes require compaction, and detects a stale sync cursor. The older C# v2 format is intentionally not loaded.

```csharp
using TurboVector;

// Create an index: 256-dimensional vectors, 4 bits per coordinate
var index = new TurboQuantIndex(dim: 256, bitWidth: 4);

// Add vectors (flat row-major float array, length = n * dim)
float[] vectors = LoadYourVectors(); // shape [n, 256]
index.Add(vectors);

// Search — returns top-k results for each query
float[] query = LoadYourQuery();     // length 256
SearchResults results = index.Search(query, k: 10);

Console.WriteLine($"Best match: slot {results.Indices[0]}, score {results.Scores[0]:F4}");

// Persist to disk (Rust-compatible v8 snapshot)
index.Write("my_index.tv");

// Reload
TurboQuantIndex loaded = TurboQuantIndex.Load("my_index.tv");
```

For stable external IDs (e.g. database row IDs):

```csharp
var idMap = new IdMapIndex(dim: 256, bitWidth: 4);
idMap.AddWithIds(vectors, ids: new ulong[] { 100, 200, 300, ... });

var (scores, ids) = idMap.Search(query, k: 5);
// ids are your original ulong keys, not slot indices

idMap.Remove(200UL); // O(1) swap-remove
idMap.Write("my_index.tvim");
```

## Algorithm Overview

| Step | What happens |
|------|--------------|
| **Rotation** | A deterministic two-round, globally permuted block-Hadamard transform driven by the pinned ChaCha8 stream. Applied once per dimension at index-build time. |
| **Codebook** | Lloyd-Max boundaries and centroids for a Beta((d−1)/2,(d−1)/2) distribution, matching the pinned Rust numerical procedure. Depends only on `bitWidth` and `dim` — reusable across indexes. |
| **Encode** | Each vector is L2-normalised → rotated → optionally TQ+ calibrated → thermometer-coded against the boundaries → bit-packed. A correction scale is stored per vector. |
| **Pack** | Codes are transposed into a blocked layout (32-vector blocks × byte-groups) for cache-friendly SIMD scoring. |
| **Search** | A uint8 look-up table maps quantized query×centroid products. The score accumulator is `scale × acc + bias`, multiplied by the per-vector correction scale. |

## Bit Width Trade-offs

| `bitWidth` | Bits/coord | Recall@10 (typical) | Index size vs float32 |
|-----------|-----------|---------------------|-----------------------|
| 2 | 2 | ~85–90 % | 1/16× |
| 3 | 3 | ~90–95 % | 3/32× |
| 4 | 4 | ~95–99 % | 1/8× |

## File Formats

| Extension | Description |
|-----------|-------------|
| `.tv` | `TurboQuantIndex` — Rust-compatible `TV8\0` v8 snapshot with calibration |
| `.tvim` | `IdMapIndex` — same v8 container with external IDs stored in each row |

## Performance Notes

The hot paths use [`System.Numerics.Tensors.TensorPrimitives`](https://learn.microsoft.com/en-us/dotnet/api/system.numerics.tensors.tensorprimitives) for JIT-vectorised SIMD:

- **Encode:** `TensorPrimitives.Norm` + `Multiply` (normalize), `TensorPrimitives.Dot` (rotation)
- **Search:** `TensorPrimitives.Dot` in `RotateQueries` (dominates query latency for large dims)

LUT score accumulation (`Score2/3/4Bit`) uses tightly-bounded scalar loops that the JIT auto-vectorises well; further optimisation with `System.Runtime.Intrinsics` (AVX-512 `VPSHUFB`) is a future opportunity.

## Project READMEs

- [Library — `src/TurboVector`](src/TurboVector/README.md)
- [Tests — `tests/TurboVector.Tests`](tests/TurboVector.Tests/README.md)
- [Examples — `examples/TurboVector.Examples`](examples/TurboVector.Examples/README.md)
- [Benchmarks — `benchmarks/TurboVector.Benchmarks`](benchmarks/TurboVector.Benchmarks/README.md)

## Credits

Ported from the Rust [`turbovec`](https://github.com/RyanCodrai/turbovec) crate by RyanCodrai, itself an implementation of the RaBitQ algorithm.
