# TurboVector — Class Library

The core algorithm library. Targets **net10.0**, enables unsafe code (bit-packing helpers), and has one NuGet dependency:

| Package | Use |
|---------|-----|
| `System.Numerics.Tensors` | SIMD-accelerated Norm, Multiply, Dot in hot paths |

Rotation, codebook construction, calibration, encoding, packing, and search are implemented in the library without external math packages.

---

## Public API

Dimensions must be positive multiples of 8 and no greater than 16,384; bit widths are 2, 3, or 4. Added vectors and search queries reject non-finite coordinates and coordinates whose magnitude is at least `1e16`. Zero vectors are accepted and receive a zero score. File writes use the Rust-compatible v8 snapshot format, a sibling temporary file, and atomic replacement. `Durability.Durable` flushes the file before replacement; `Durability.Fast` skips the forced flush. `Sync(path)` is the durable stateful API: it appends new complete 32-row units and commits an alternating v8 header when the existing cursor is current. Changes that cannot be represented as append-only data (such as swap-removal or recalibration) use an atomic full rewrite. A path bound by `Sync` or loaded from a claimed v8 file is protected against replacement or advancement by another writer. Both index types also support `ToBytes()` and `WriteTo(Stream)`; loading is available from a path, stream, or byte span. Rust v7 and v8 images are accepted; the former C# v2 format is intentionally not loaded.

### `TurboQuantIndex` — slot-based index

```csharp
// Eager construction (dimension known upfront)
var index = new TurboQuantIndex(int dim, int bitWidth);

// Lazy construction (dimension inferred from first batch)
var index = TurboQuantIndex.NewLazy(int bitWidth);

// Add vectors (flat row-major, length must be a multiple of dim)
index.Add(ReadOnlySpan<float> vectors);
index.Add2D(ReadOnlySpan<float> vectors, int dim); // also sets dim on lazy index

// Optional TQ+ per-coordinate calibration from a representative sample
index.Calibrate(ReadOnlySpan<float> sample);
index.Calibrate2D(ReadOnlySpan<float> sample, int dim); // also sets dim on lazy index

// Search — returns top-k results for every query row
SearchResults results = index.Search(ReadOnlySpan<float> queries, int k);

// Search with a per-slot boolean inclusion mask
SearchResults results = index.SearchWithMask(ReadOnlySpan<float> queries, int k, ReadOnlySpan<bool> mask);

// Remove a vector by slot index (O(1) swap-remove; returns the slot that was swapped in)
int movedSlot = index.SwapRemove(int idx);

// Warm up lazy caches (optional — saves first-search latency)
index.Prepare();

// Persist / reload
index.Write(string path);                          // durable v8 snapshot
index.Write(string path, Durability.Fast);         // atomic, without forced disk flush
index.Sync(string path);                           // durable append-oriented sync
byte[] snapshot = index.ToBytes();
index.WriteTo(Stream output);
TurboQuantIndex loaded = TurboQuantIndex.Load(string path);
TurboQuantIndex fromBytes = TurboQuantIndex.Load(snapshot);

// Properties
int  index.Len       // number of vectors currently stored
bool index.IsEmpty
int  index.Dim       // 0 if lazy and no vectors added yet
int? index.DimOpt    // null if lazy and no vectors added yet
int  index.BitWidth
CalibrationState index.CalibrationState
```

Calibration adjusts quantization to a representative sample and applies the matching inverse transform and bias correction during search. Its shift and scale arrays are persisted in the v8 superblock and restored when loading.

### `SearchResults`

```csharp
float[] results.Scores    // flat array, length = nq × k
long[]  results.Indices   // flat array, length = nq × k (slot indices)
int     results.Nq        // number of queries
int     results.K         // effective k (may be < requested k if index is small)

// Slice helpers for multi-query results
ReadOnlySpan<float> results.ScoresForQuery(int qi)
ReadOnlySpan<long>  results.IndicesForQuery(int qi)
```

### `IdMapIndex` — stable external-ID wrapper

Wraps `TurboQuantIndex` and maintains a bidirectional `ulong↔slot` map. Useful when your vectors have stable external IDs (e.g. database primary keys).

```csharp
var idMap = new IdMapIndex(int dim, int bitWidth);
var idMap = IdMapIndex.NewLazy(int bitWidth);

// Add with IDs — ids.Length must equal vectors.Length / dim
idMap.AddWithIds(ReadOnlySpan<float> vectors, ReadOnlySpan<ulong> ids);
idMap.AddWithIds2D(ReadOnlySpan<float> vectors, int dim, ReadOnlySpan<ulong> ids);

// Optional calibration is shared with the wrapped slot-based index
idMap.Calibrate(ReadOnlySpan<float> sample);
idMap.Calibrate2D(ReadOnlySpan<float> sample, int dim);

// Remove by external ID (O(1)); returns false if not found
bool removed = idMap.Remove(ulong id);

// Search — returns scores and your original ulong IDs
(float[] Scores, ulong[] Ids) = idMap.Search(ReadOnlySpan<float> queries, int k);

// Search restricted to a specific set of IDs
// allowlist == null → search all; allowlist.Length == 0 → throws
(float[] Scores, ulong[] Ids) = idMap.SearchWithAllowlist(queries, k, ulong[]? allowlist);

bool  idMap.Contains(ulong id)
int   idMap.Len
bool  idMap.IsEmpty
int   idMap.Dim
int?  idMap.DimOpt
int   idMap.BitWidth

idMap.Write(string path);                    // durable v8 snapshot
idMap.Sync(string path);                     // durable append-oriented sync
byte[] snapshot = idMap.ToBytes();
idMap.WriteTo(Stream output);
IdMapIndex loaded = IdMapIndex.Load(string path);
```

---

## Internal Modules

| File | Responsibility |
|------|----------------|
| `Constants.cs` | `Block=32`, `FlushEvery=256`, `RotationSeed=42UL` |
| `Rotation.cs` | Deterministic two-round globally permuted block-Hadamard transform, with scalar and AVX2 kernels |
| `Codebook.cs` | Lloyd-Max quantizer for Beta((d−1)/2,(d−1)/2), using the pinned upstream Beta and adaptive-Simpson algorithms |
| `TqPlusCalibration.cs` | Fits deterministic per-coordinate shift/scale values from a representative sample |
| `Encode.cs` | Normalize → rotate → calibrate → thermometer-code → pack → compute correction scales |
| `Pack.cs` | Transpose packed codes into SIMD-friendly blocked layout (32-vector blocks) |
| `Search.cs` | LUT-based scored search for 2-bit, 3-bit, and 4-bit codes; min-heap top-k |
| `Io.cs`, `PersistenceV8.cs` | Rust-compatible v8 snapshots, v7/v8 loading, calibration, stream and byte APIs |

### `Encoder.Encode` pipeline (hot path)

```
vectors (float[], n×dim)
   │
   ├─ Loop A: TensorPrimitives.Norm + Multiply  → unit (L2-normalised rows)
   ├─ Loop B: TensorPrimitives.Dot per row      → rotated (after orthogonal rotation)
   ├─ Loop C: thermometer-code vs boundaries    → codes (byte, 0…levels-1)
   ├─ Loop D: dot(rotated, centroid[codes])     → correction scales
   └─ PackCodes: bit-plane packing             → packed (byte[], n × bitWidth × dim/8)
```

### `Search.RotateQueries` (hottest query-time path)

```
queries (float[], nq×dim)
   │
   └─ TensorPrimitives.Dot per row × dim       → rotated queries (SIMD, 4–8× vs scalar)
```

### Scoring kernel (per candidate vector)

```
acc = Σ_g lut[g_offset + nibble(packed[g])]   // LUT lookup, byte accumulation
score = scale * acc + bias                     // dequantise
score *= vecScale[vecIdx]                      // per-vector correction
```

---

## File Format Details

### `.tv` and `.tvim` (Rust-compatible v8)

```
superblock      TV8\0, revision, bit width, index kind, unit layout, dimension,
                file nonce, redo-op capacity, canonical codebook, TQ+ calibration,
                CRC-32C
header slots    Two alternating generation/count/tail/redo-op/delta-digest headers
block units     32 rows of sequential blocked codes, scales, and optional IDs
```

`Write`, `ToBytes`, and `WriteTo` emit v8 sequential-layout snapshots with an unclaimed nonce; readers accept Rust v7 sequential files and v8 sequential or planes-layout files. `Sync` claims its path with a random nonce, flushes appended units before publishing the alternate commit header, and verifies the previous header's delta before advancing it. Incomplete appends are ignored by the last valid header and truncated before retry. The file kind distinguishes slot indexes from ID-mapped indexes. `Durability.Durable` flushes snapshot contents before atomic replacement; `Sync` is always durable.

---

## Error Reference

| Condition | Exception type | Message substring |
|-----------|---------------|-------------------|
| `Add()` on lazy index (dim not set) | `InvalidOperationException` | `"dim is not set"` |
| `Add2D()` dim mismatch | `ArgumentException` | `"dim mismatch"` |
| `SwapRemove()` out of range | `ArgumentOutOfRangeException` | `"out of bounds"` |
| `SearchWithMask()` wrong mask length | `ArgumentException` | `"mask length"` |
| `AddWithIds()` on lazy index | `InvalidOperationException` | `"dim is not set"` |
| `SearchWithAllowlist()` empty allowlist | `ArgumentException` | `"allowlist is empty"` |
| `Io.Load()` v1 file | `InvalidDataException` | `"turbovec ≤ 0.4.3"` + `"Rebuild"` |
| `Io.Load()` wrong magic | `InvalidDataException` | `"wrong magic"` |
