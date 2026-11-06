# TurboVector.Examples

The console app keeps small end-to-end usage samples and provides named commands for direct .NET equivalents of selected upstream `turbovec` examples. Benchmark commands are opt-in; their default workloads are intentionally smaller than the upstream stress-test defaults.

## Running

Run the existing basic usage and ID-map samples:

```bash
dotnet run --project examples/TurboVector.Examples
```

List commands:

```bash
dotnet run --project examples/TurboVector.Examples -- help
```

Run an individual sample:

```bash
dotnet run --project examples/TurboVector.Examples -- encode-hash
dotnet run --project examples/TurboVector.Examples -- kernel-xtest --dim 512 --bits 4 --n-db 1000 --n-queries 8 --k 10
dotnet run --project examples/TurboVector.Examples -- insert-bench --dim 128 --bits 2 --n 5000
dotnet run --project examples/TurboVector.Examples -- sync-bench --dim 128 --bits 2 --n 10000
dotnet run --project examples/TurboVector.Examples -- stream-bw --small-mib 32 --large-mib 128
dotnet run --project examples/TurboVector.Examples -- --dump-state ./state_output
```

## Examples

| Command | Upstream source | Purpose |
|---------|----------------|---------|
| no arguments | Existing C# samples | Basic search, stable IDs, filtering, and snapshot round trips |
| `encode-hash` | `encode_hash.rs` | Print FNV-1a fingerprints for codebook, calibration, packed codes, scales, and snapshot bytes across the same deterministic dimension/bit-width cells used by the C# parity tests |
| `kernel-xtest` | `kernel_xtest.rs` | Build deterministic normalized data, run top-K search, and print each query's sorted result-ID set for cross-platform comparison; scores and within-top-K ordering are intentionally omitted |
| `insert-bench` | `insert_bench.rs` | Report median bulk/warm insert throughput and single-add/ID-map-remove/swap-remove latency; initial bulk timing includes explicit calibration |
| `sync-bench` | `sync_bench_v7.rs` | Measure full sync, complete-unit appends, and sync after removals using the C# v8 persistence API; removal sync currently measures compaction/full rewrite |
| `stream-bw` | `stream_bw.rs` | Estimate sequential memory-read bandwidth with multithreaded and single-threaded scans |
| `--dump-state [path]` | Existing C# utility | Write rotation, boundaries, and centroids as little-endian float arrays to `state_d*_b*.bin` files |

`insert-bench` accepts `--dim`, `--bits`, `--n`, `--append`, `--singles`, `--removes`, and `--repeats`. `sync-bench` accepts `--dim`, `--bits`, `--n`, `--append-repeats`, `--removes`, and `--scattered`. `stream-bw` accepts `--small-mib`, `--large-mib`, and `--threads`. The `kernel-xtest` options are `--dim`, `--bits`, `--seed`, `--n-db`, `--n-queries`, and `--k`.

## Upstream examples not ported

The current upstream set also includes `convert.rs`, `kernel_roofline.rs`, `kernel_roofline_neon.rs`, `load_bench_v7.rs`, `probe_2bit_lutstream.rs`, `probe_2bit_sdot.rs`, `probe_2bit_vnni.rs`, `sve_tbl_probe.rs`, `vector_major_check.rs`, `vnni_fullscan.rs`, `vnni_probe.rs`, and `x86_permute_dot.rs`. These are not direct equivalents of the current public C# API: format conversion/old-format comparisons target unsupported versions, kernel probes depend on Rust-only or ISA-specific search internals, and vector-major/VNNI experiments measure layouts not used by this implementation. They are intentionally excluded rather than replaced with benchmarks that would measure different work.

## Existing usage samples

### `BasicUsage`

Creates 100 normalized 64-dimensional vectors, adds them to a 4-bit index, verifies self-search, prints the top five results, writes a temporary `.tv` snapshot, and checks that loading preserves the top result.

### `IdMapExample`

Adds vectors with stable external IDs, searches and removes by ID, exercises allowlist filtering, and verifies a `.tvim` round trip.

### `DumpState`

Writes one binary state file for each configured dimension/bit-width pair. Each file contains the dense rotation matrix followed by codebook boundaries and centroids as little-endian `float32` values; it does not write index snapshots.
