# TurboVector.Examples

A console application demonstrating common TurboVector usage patterns.

## Running

```bash
# From solution root — basic usage + IdMap example
dotnet run --project examples/TurboVector.Examples

# Also run the DumpState example (writes .tv/.tvim files to a directory)
dotnet run --project examples/TurboVector.Examples -- --dump-state ./state_output
```

## Examples

### `BasicUsage` — `BasicUsage.cs`

Demonstrates the simplest end-to-end workflow with `TurboQuantIndex`:

1. Generate 100 normalised random vectors of dimension 64.
2. Create a 4-bit index and add all vectors.
3. Search for the 5 nearest neighbours of vector 0.
4. Assert that vector 0 is its own top match (self-score ≈ 1.0).
5. Persist the index to a temp `.tv` file with `index.Write()`.
6. Reload with `TurboQuantIndex.Load()` and verify the top result is unchanged.

**Expected output:**
```
=== Basic Usage ===
Indexed 100 vectors (dim=64, bits=4).
Top result for vector 0: slot=0, score=0.9997
Top-5 results:
  1. slot=0, score=0.9997
  2. slot=42, score=0.7123
  ...
Reloaded index top result: slot=0, score=0.9997
```

### `IdMapExample` — `IdMapExample.cs`

Demonstrates `IdMapIndex` with stable external IDs:

1. Create a lazy `IdMapIndex` (bit-width 3, dimension inferred from first batch).
2. Add three batches of vectors with `ulong` IDs (100, 200, 300, …).
3. Search and display results as your original IDs, not slot numbers.
4. Remove a vector by ID with `idMap.Remove(id)`.
5. Verify the removed ID no longer appears in search results.
6. Persist/reload with `.tvim` round-trip.
7. Demonstrate `SearchWithAllowlist` — restricting search to a subset of IDs.

### `DumpState` — `DumpState.cs`

Utility example that writes both a `.tv` and a `.tvim` file to a specified directory. Useful for inspecting the binary format or generating test fixtures.

```bash
dotnet run --project examples/TurboVector.Examples -- --dump-state ./my_output
# Creates:
#   ./my_output/basic.tv
#   ./my_output/idmap.tvim
```

## Key Patterns Illustrated

| Pattern | Example |
|---------|---------|
| Eager index (dim known) | `new TurboQuantIndex(dim, bitWidth)` |
| Lazy index (dim from data) | `TurboQuantIndex.NewLazy(bitWidth)` / `index.Add2D(vectors, dim)` |
| Stable external IDs | `IdMapIndex.AddWithIds(vectors, ids)` |
| Filtered search | `IdMapIndex.SearchWithAllowlist(query, k, allowlist)` |
| Persistence | `index.Write(path)` / `TurboQuantIndex.Load(path)` |
| Incremental removal | `IdMapIndex.Remove(id)` (O(1) swap-remove) |
