using Hasseware.Extensions.VectorData.TurboQuant;
using Microsoft.Extensions.VectorData;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using TurboVector;

namespace TurboVector.NativeAot.Smoke;

internal static class Program
{
    private const ulong FnvOffset = 0xcbf29ce484222325;
    private const ulong FnvPrime = 0x100000001b3;

    private static async Task Main()
    {
        Require(!RuntimeFeature.IsDynamicCodeSupported, "The smoke test must run as a Native AOT executable.");
        Require(!JsonSerializer.IsReflectionEnabledByDefault, "Reflection-based JSON serialization must be disabled.");

        CheckCore();
        await CheckProviderAsync();
        Console.WriteLine("Native AOT smoke test passed.");
    }

    private static void CheckCore()
    {
        const int dimension = 200;
        const int bitWidth = 2;
        const int rows = 2000;
        float[] vectors = CreateVectors(rows, dimension, bitWidth);
        var index = new TurboQuantIndex(dimension, bitWidth);
        index.Calibrate(vectors);
        index.Add(vectors);

        Require(index.CalibrationState == CalibrationState.Calibrated, "Calibration was not retained.");
        Require(Hash(index.TqPlusScale, Hash(index.TqPlusShift)) == 0x1974331dac161d9b, "Calibration differs from the Rust fixture.");
        Require(Hash(index.PackedCodes) == 0x122b777da643eba6, "Quantized codes differ from the Rust fixture.");
        Require(Hash(index.Scales) == 0x467e7fc11cbd63c8, "Quantization scales differ from the Rust fixture.");

        SearchResults before = index.Search(vectors.AsSpan(0, dimension), 3);
        Require(before.Indices.Length == 3 && before.Indices[0] == 0, "Core search did not return the matching vector.");

        byte[] image = index.ToBytes();
        Require(image.AsSpan(0, 4).SequenceEqual("TV8\0"u8), "Core persistence did not produce a v8 image.");
        TurboQuantIndex restored = TurboQuantIndex.Load(image);
        SearchResults after = restored.Search(vectors.AsSpan(0, dimension), 3);
        Require(index.PackedCodes.SequenceEqual(restored.PackedCodes), "Core persistence changed quantized codes.");
        Require(index.Scales.SequenceEqual(restored.Scales), "Core persistence changed quantization scales.");
        Require(index.TqPlusShift.SequenceEqual(restored.TqPlusShift), "Core persistence changed calibration shifts.");
        Require(index.TqPlusScale.SequenceEqual(restored.TqPlusScale), "Core persistence changed calibration scales.");
        Require(before.Indices.AsSpan().SequenceEqual(after.Indices), "Core persistence changed search results.");
        Require(before.Scores.AsSpan().SequenceEqual(after.Scores), "Core persistence changed search scores.");
    }

    private static async Task CheckProviderAsync()
    {
        string directory = Path.Combine(Path.GetTempPath(), $"turbovector-aot-{Guid.NewGuid():N}");
        try
        {
            var options = new JsonSerializerOptions { TypeInfoResolver = SmokeJsonContext.Default };
            var store = new TurboQuantFileSystemVectorStore(directory, serializerOptions: options);
            var mapping = new TurboQuantRecordMapping<string, SmokeDocument>
            {
                Definition = new VectorStoreCollectionDefinition
                {
                    Properties =
                    [
                        new VectorStoreKeyProperty(nameof(SmokeDocument.Id), typeof(string)),
                        new VectorStoreDataProperty(nameof(SmokeDocument.Category), typeof(string)) { StorageName = "category" },
                        new VectorStoreDataProperty(nameof(SmokeDocument.Rank), typeof(int)),
                        new VectorStoreVectorProperty(nameof(SmokeDocument.Embedding), typeof(ReadOnlyMemory<float>), 8)
                    ]
                },
                RecordTypeInfo = SmokeJsonContext.Default.SmokeDocument,
                KeyTypeInfo = SmokeJsonContext.Default.String,
                GetKey = static document => document.Id,
                GetVector = static document => document.Embedding
            };
            float[] first = [1, 0, 0, 0, 0, 0, 0, 0];
            float[] second = [0, 1, 0, 0, 0, 0, 0, 0];
            TurboQuantCollection<string, SmokeDocument> documents = store.GetCollection("documents", mapping);
            await documents.UpsertAsync(
            [
                new SmokeDocument { Id = "a", Category = "manual", Rank = 1, Embedding = first },
                new SmokeDocument { Id = "b", Category = "guide", Rank = 2, Embedding = second }
            ]);

            SmokeDocument? full = await documents.GetAsync("b", new RecordRetrievalOptions { IncludeVectors = true });
            Require(full is not null && full.Embedding.Span.SequenceEqual(second), "Typed record vectors did not round-trip.");
            SmokeDocument? withoutVector = await documents.GetAsync("b");
            Require(withoutVector is not null && withoutVector.Category == "guide" && withoutVector.Embedding.IsEmpty,
                "Typed record retrieval without vectors failed.");

            var typedResults = new List<VectorSearchResult<SmokeDocument>>();
            await foreach (VectorSearchResult<SmokeDocument> result in documents.SearchAsync(
                new ReadOnlyMemory<float>(second),
                top: 1,
                new VectorSearchOptions<SmokeDocument> { Filter = document => document.Category == "guide", IncludeVectors = true }))
            {
                typedResults.Add(result);
            }

            Require(typedResults.Count == 1 && typedResults[0].Record.Id == "b" &&
                typedResults[0].Record.Embedding.Span.SequenceEqual(second), "Typed filtered search failed.");

            var ordered = new List<SmokeDocument>();
            await foreach (SmokeDocument record in documents.GetAsync(
                document => document.Rank >= 1,
                top: 1,
                new FilteredRecordRetrievalOptions<SmokeDocument>
                {
                    OrderBy = order => order.Descending(document => document.Rank)
                }))
            {
                ordered.Add(record);
            }

            Require(ordered.Count == 1 && ordered[0].Id == "b", "Typed filtered ordering failed.");

            var dynamicDefinition = new VectorStoreCollectionDefinition
            {
                Properties =
                [
                    new VectorStoreKeyProperty("Id", typeof(string)),
                    new VectorStoreDataProperty("Rank", typeof(int)),
                    new VectorStoreDataProperty("Details", typeof(SmokeDetails)),
                    new VectorStoreVectorProperty("Embedding", typeof(ReadOnlyMemory<float>), 8)
                ]
            };
            VectorStoreCollection<object, Dictionary<string, object?>> dynamicRecords =
                store.GetDynamicCollection("dynamic", dynamicDefinition);
            await dynamicRecords.UpsertAsync(new Dictionary<string, object?>
            {
                ["Id"] = "dynamic-a",
                ["Rank"] = 3,
                ["Details"] = new SmokeDetails { Title = "notes" },
                ["Embedding"] = new ReadOnlyMemory<float>(first)
            });

            Dictionary<string, object?>? dynamicRecord = await dynamicRecords.GetAsync(
                "dynamic-a", new RecordRetrievalOptions { IncludeVectors = true });
            Require(dynamicRecord is not null &&
                dynamicRecord["Rank"] is 3 &&
                dynamicRecord["Details"] is SmokeDetails { Title: "notes" } &&
                dynamicRecord["Embedding"] is ReadOnlyMemory<float> embedding && embedding.Span.SequenceEqual(first),
                "Dynamic record values did not round-trip.");

            var dynamicResults = new List<VectorSearchResult<Dictionary<string, object?>>>();
            await foreach (VectorSearchResult<Dictionary<string, object?>> result in dynamicRecords.SearchAsync(
                new ReadOnlyMemory<float>(first),
                top: 1,
                new VectorSearchOptions<Dictionary<string, object?>> { Filter = record => (int)record["Rank"]! >= 3 }))
            {
                dynamicResults.Add(result);
            }

            Require(dynamicResults.Count == 1 &&
                Equals(dynamicResults[0].Record["Id"], "dynamic-a") &&
                !dynamicResults[0].Record.ContainsKey("Embedding"), "Dynamic filtered search failed.");

            var reopened = new TurboQuantFileSystemVectorStore(directory, serializerOptions: options);
            Require((await reopened.GetCollection("documents", mapping).GetAsync("b"))?.Rank == 2,
                "Typed collection could not be reopened.");
            Dictionary<string, object?>? reopenedDynamic = await reopened.GetDynamicCollection("dynamic", dynamicDefinition).GetAsync("dynamic-a");
            Require(reopenedDynamic is not null && reopenedDynamic["Details"] is SmokeDetails { Title: "notes" } &&
                !reopenedDynamic.ContainsKey("Embedding"), "Dynamic collection could not be reopened.");

            var names = new List<string>();
            await foreach (string name in reopened.ListCollectionNamesAsync())
            {
                names.Add(name);
            }

            Require(names.SequenceEqual(["documents", "dynamic"]), "Persisted collection listing failed.");
            await reopened.EnsureCollectionDeletedAsync("dynamic");
            Require(!await reopened.CollectionExistsAsync("dynamic"), "Collection deletion failed.");
        }
        finally
        {
            if (Directory.Exists(directory))
            {
                Directory.Delete(directory, recursive: true);
            }
        }
    }

    private static float[] CreateVectors(int rows, int dimension, int bitWidth)
    {
        float[] vectors = new float[checked(rows * dimension)];
        ulong state = (0x7B0_0000UL ^ ((ulong)dimension << 8) ^ (uint)bitWidth) | 1UL;
        for (int i = 0; i < vectors.Length; i++)
        {
            state = unchecked(state * 6_364_136_223_846_793_005UL + 1_442_695_040_888_963_407UL);
            vectors[i] = (float)((uint)(state >> 32) / 2_147_483_648.0 - 1.0);
        }

        return vectors;
    }

    private static ulong Hash(ReadOnlySpan<byte> values)
    {
        ulong hash = FnvOffset;
        foreach (byte value in values)
        {
            hash = unchecked((hash ^ value) * FnvPrime);
        }

        return hash;
    }

    private static ulong Hash(ReadOnlySpan<float> values, ulong hash = FnvOffset)
    {
        foreach (float value in values)
        {
            uint bits = BitConverter.SingleToUInt32Bits(value);
            for (int shift = 0; shift < 32; shift += 8)
            {
                hash = unchecked((hash ^ (byte)(bits >> shift)) * FnvPrime);
            }
        }

        return hash;
    }

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }
}

internal sealed class SmokeDocument
{
    public string Id { get; set; } = "";

    [JsonPropertyName("category")]
    public string Category { get; set; } = "";

    public int Rank { get; set; }

    public ReadOnlyMemory<float> Embedding { get; set; }
}

internal sealed class SmokeDetails
{
    public string Title { get; set; } = "";
}

[JsonSerializable(typeof(SmokeDocument))]
[JsonSerializable(typeof(SmokeDetails))]
[JsonSerializable(typeof(string))]
internal sealed partial class SmokeJsonContext : JsonSerializerContext;
