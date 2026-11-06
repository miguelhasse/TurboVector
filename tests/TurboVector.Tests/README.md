# TurboVector.Tests

xUnit.v3 unit and integration test suite. **84 tests, 0 skipped.**

## Running Tests

```bash
# From solution root
dotnet test tests/TurboVector.Tests

# Verbose output
dotnet test tests/TurboVector.Tests --logger "console;verbosity=normal"

# Run a specific test class
dotnet test tests/TurboVector.Tests --filter "ClassName=TurboVector.Tests.SearchTests"

# Run a specific test by name
dotnet test tests/TurboVector.Tests --filter "DisplayName~Codebook"
```

## Test Files

| File | Tests | What's covered |
|------|-------|----------------|
| `CodebookTests.cs` | Boundary count, centroid ordering, reproducibility, dimension scaling | Lloyd-Max quantizer (`Codebook.Compute`) |
| `RotationTests.cs` | Orthogonality (‖Rx‖≈‖x‖), determinism, different dims | `Rotation.MakeRotationMatrix` |
| `EncodeTests.cs` | Packed code dimensions, scale positivity, self-score ≈ 1.0 | `Encoder.Encode` end-to-end |
| `KernelCorrectnessTests.cs` | 2-bit/3-bit/4-bit search recall, score ordering | `Search.RunSearch`, `Search_`, `Search3Bit_` |
| `DistortionTests.cs` | Mean distortion bounds for 2/3/4-bit at dim 64/256 | Full encode→search pipeline |
| `FilteringTests.cs` | Mask-filtered search returns only allowed slots; `BlocksSkippedByMask` increments | `TurboQuantIndex.SearchWithMask` |
| `IdMapTests.cs` | Add, search, remove, allowlist, duplicate ID rejection, persistence | `IdMapIndex` full API |
| `IoVersioningTests.cs` | Round-trip `.tv`/`.tvim`, v1 rejection, wrong-magic rejection | `Io` static class |
| `SwapRemoveTests.cs` | Swap-remove semantics, index shrinkage, OOB error | `TurboQuantIndex.SwapRemove` |
| `LazyInitTests.cs` | Lazy index dimension commitment, error before dim is set | `TurboQuantIndex.NewLazy`, `IdMapIndex.NewLazy` |
| `ConcurrentSearchTests.cs` | Parallel search from multiple threads without data races | `TurboQuantIndex.Search` thread safety |

## Shared Test Helpers (`CodebookTests.cs — TestHelpers`)

All test files share deterministic PRNG helpers defined as static methods in `TestHelpers`:

| Helper | Algorithm | Used by |
|--------|-----------|---------|
| `MakeVectors(n, dim, seed?)` | LCG (`state = state×6364136223846793005 + 1442695040888963407`), float from mantissa bits | Most test files |
| `GaussianNormalized(n, dim, seed?)` | xorshift64 + Box-Muller + L2 normalize | DistortionTests, KernelCorrectnessTests |
| `UnitSphereVectors(n, dim, seed?)` | Same LCG + Box-Muller + L2 normalize | FilteringTests |
| `RandVec(dim, ref state)` | Single LCG step → float ∈ [−1, 1] | Internal helper |

These helpers deliberately match the PRNGs used in the original Rust test suite so that expected values (distortion thresholds, recall rates) remain comparable.

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `xunit.v3` | 1.1.0 | Test framework |
| `xunit.v3.runner.console` | 1.1.0 | Console runner |
| `Microsoft.NET.Test.Sdk` | latest | MSBuild integration |
| `MathNet.Numerics` | 5.0.0 | Beta distribution in `DistortionTests.cs` only |

## Conventions

- Each test class exercises one public class or module.
- Error message assertions check for **substrings** (not exact equality) to tolerate minor wording changes.
- Distortion and recall tests use generous tolerances (±5–10 %) to remain stable across floating-point differences between platforms.
- `ConcurrentSearchTests` runs 8 threads × 200 iterations and asserts that all results agree with a single-threaded reference run.
