# TurboVector — Class Library

The core algorithm library. Targets **net10.0**, enables unsafe code (bit-packing helpers), and has one NuGet dependency:

| Package | Use |
|---------|-----|
| `System.Numerics.Tensors` | SIMD-accelerated Norm, Multiply, Dot in hot paths |

All mathematical routines (Householder QR, Lanczos LogGamma, Lentz continued-fraction Beta CDF, Gauss-Legendre quadrature) are implemented natively — no external math library is required.

---

## Public API

### `TurboQuantIndex` — slot-based index

```csharp
// Eager construction (dimension known upfront)
var index = new TurboQuantIndex(int dim, int bitWidth);

// Lazy construction (dimension inferred from first batch)
var index = TurboQuantIndex.NewLazy(int bitWidth);

// Add vectors (flat row-major, length must be a multiple of dim)
index.Add(ReadOnlySpan<float> vectors);
index.Add2D(ReadOnlySpan<float> vectors, int dim); // also sets dim on lazy index

// Search — returns top-k results for every query row
SearchResults results = index.Search(ReadOnlySpan<float> queries, int k);

// Search with a per-slot boolean inclusion mask
SearchResults results = index.SearchWithMask(ReadOnlySpan<float> queries, int k, ReadOnlySpan<bool> mask);

// Remove a vector by slot index (O(1) swap-remove; returns the slot that was swapped in)
int movedSlot = index.SwapRemove(int idx);

// Warm up lazy caches (optional — saves first-search latency)
index.Prepare();

// Persist / reload
index.Write(string path);                          // writes .tv file
TurboQuantIndex loaded = TurboQuantIndex.Load(string path);

// Properties
int  index.Len       // number of vectors currently stored
bool index.IsEmpty
int  index.Dim       // 0 if lazy and no vectors added yet
int? index.DimOpt    // null if lazy and no vectors added yet
int  index.BitWidth
```

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

idMap.Write(string path);                    // writes .tvim file
IdMapIndex loaded = IdMapIndex.Load(string path);
```

---

## Internal Modules

| File | Responsibility |
|------|----------------|
| `Constants.cs` | `Block=32`, `FlushEvery=256`, `RotationSeed=42UL` |
| `Rotation.cs` | Deterministic orthogonal matrix via ChaCha8 RNG + native Householder QR; includes `ChaCha8Rng` (8-round) and Box-Muller Gaussian sampling |
| `Codebook.cs` | Lloyd-Max quantizer — boundaries and centroids for Beta((d−1)/2,(d−1)/2); native Beta CDF via Lanczos LogGamma + Lentz continued-fraction; Gauss-Legendre quadrature fallback |
| `Encode.cs` | Normalise → rotate → thermometer-code → pack → compute correction scales |
| `Pack.cs` | Transpose packed codes into SIMD-friendly blocked layout (32-vector blocks) |
| `Search.cs` | LUT-based scored search for 2-bit, 3-bit, and 4-bit codes; min-heap top-k |
| `Io.cs` | Binary `.tv` (v2) and `.tvim` serialisation/deserialisation |

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

### `.tv` (TurboQuantIndex v2)

```
Bytes   Field
4       magic: ASCII "TVPI"
1       version: 0x02
1       bit_width (u8)
4       dim (u32 LE)
4       n_vectors (u32 LE)
n_vectors × (dim/8 × bit_width)   packed_codes (byte[])
n_vectors × 4                      scales (float32 LE[])
```

v1 detection: first byte ∈ {2, 3, 4} with no `TVPI` magic → `InvalidDataException` with a rebuild hint.

### `.tvim` (IdMapIndex v2)

Same header as `.tv` but magic `TVIM`, followed by `n_vectors × 8` bytes of `slot_to_id` (ulong LE[]).

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
