# TurboVector.Benchmarks

BenchmarkDotNet performance harness for the TurboVector library.

> **Important:** Always run benchmarks in `Release` configuration. Debug builds disable the JIT optimisations that TensorPrimitives and the scoring loops depend on.

## Running Benchmarks

```bash
# From solution root — run all benchmarks
dotnet run --project benchmarks/TurboVector.Benchmarks -c Release

# Filter to a specific benchmark class
dotnet run --project benchmarks/TurboVector.Benchmarks -c Release -- --filter "*Search*"
dotnet run --project benchmarks/TurboVector.Benchmarks -c Release -- --filter "*Encode*"
dotnet run --project benchmarks/TurboVector.Benchmarks -c Release -- --filter "*IdMap*"
dotnet run --project benchmarks/TurboVector.Benchmarks -c Release -- --filter "*Filtered*"

# Export results to CSV / HTML / JSON
dotnet run --project benchmarks/TurboVector.Benchmarks -c Release -- --exporters csv html

# List available benchmarks without running
dotnet run --project benchmarks/TurboVector.Benchmarks -c Release -- --list flat
```

Results are written to `BenchmarkDotNet.Artifacts/` under the project directory.

## Benchmark Classes

### `EncodeBenchmarks` — `EncodeBenchmarks.cs`

Measures the full encode pipeline (`Encoder.Encode`) including normalisation, rotation, thermometer-coding, and bit-packing.

| Parameter | Values |
|-----------|--------|
| `BitWidth` | 2, 4 |
| `Dim` | 256, 1536 |
| `N` (fixed) | 1 000 vectors |

**Benchmark:**
- `Encode` — encodes 1 000 vectors with pre-computed rotation and codebook.

Use this to measure the impact of TensorPrimitives on Loops A (normalise) and B (rotate).

---

### `SearchBenchmarks` — `SearchBenchmarks.cs`

Measures nearest-neighbour search latency against varying index sizes.

| Parameter | Values |
|-----------|--------|
| `BitWidth` | 2, 4 |
| `Dim` | 1536 |
| `N` | 1 000, 10 000, 100 000 |

**Benchmarks:**
- `SearchSingleQuery` — one query, k=10.
- `SearchBatch` — 100 sequential queries, each k=10.

The `RotateQueries` call dominates for large `Dim`; this is where `TensorPrimitives.Dot` gives the most benefit.

---

### `FilteredSearchBenchmarks` — `FilteredSearchBenchmarks.cs`

Measures the effect of boolean-mask filtering on search throughput.

| Parameter | Values |
|-----------|--------|
| `BitWidth` | 4 |
| `Dim` | 256 |
| `N` | 10 000 |
| `FractionAllowed` | 0.1, 0.5, 1.0 |

**Benchmarks:**
- `SearchWithMask` — search with fraction of vectors marked as allowed.
- `SearchNoMask` — baseline unfiltered search.

Use `Search.BlocksSkippedByMask` counter (visible in examples) to measure block-skip efficiency.

---

### `IdMapSearchBenchmarks` — `IdMapSearchBenchmarks.cs`

Measures `IdMapIndex` overhead vs raw `TurboQuantIndex`.

| Parameter | Values |
|-----------|--------|
| `BitWidth` | 4 |
| `Dim` | 256 |
| `N` | 10 000 |

**Benchmarks:**
- `SearchRaw` — baseline `TurboQuantIndex.Search`.
- `SearchIdMap` — `IdMapIndex.Search` (includes ID remapping).
- `SearchAllowlist` — `IdMapIndex.SearchWithAllowlist` with 50 % of IDs allowed.

---

## Runtime Note

Benchmarks use `[SimpleJob(RuntimeMoniker.HostProcess)]` rather than `RuntimeMoniker.Net100` because BenchmarkDotNet 0.14.0 does not yet enumerate .NET 10 as a named moniker. The host process is .NET 10, so results are accurate.

## Interpreting Results

| Column | Meaning |
|--------|---------|
| `Mean` | Average execution time (ns / μs / ms) |
| `StdDev` | Standard deviation |
| `Ratio` | Relative to baseline (when `[Benchmark(Baseline=true)]` is set) |
| `Alloc` | Managed heap bytes allocated per operation |
| `Gen0` / `Gen1` | GC collection rate |

For search benchmarks, `Alloc > 0` reflects the heap allocation of the result arrays (`float[]`, `long[]`). Hot encode paths allocate rotation + codebook once in `GlobalSetup` and reuse them.

## Dependencies

| Package | Version |
|---------|---------|
| `BenchmarkDotNet` | 0.14.0 |
