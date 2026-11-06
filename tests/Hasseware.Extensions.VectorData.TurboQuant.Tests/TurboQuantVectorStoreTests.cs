using Microsoft.Extensions.VectorData;
using Microsoft.Extensions.VectorData.ProviderServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using TurboVector;

namespace Hasseware.Extensions.VectorData.TurboQuant.Tests;

public sealed class TurboQuantVectorStoreTests : IDisposable
{
    private static readonly float[] Axis0 = [1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f];
    private static readonly float[] Axis1 = [0f, 1f, 0f, 0f, 0f, 0f, 0f, 0f];
    private static readonly float[] Axis2 = [0f, 0f, 1f, 0f, 0f, 0f, 0f, 0f];

    private readonly string _directory = Path.Combine(Path.GetTempPath(), $"tq-vectordata-{Guid.NewGuid():N}");

    public void Dispose()
    {
        if (Directory.Exists(_directory))
        {
            Directory.Delete(_directory, recursive: true);
        }
    }

    [Fact]
    public async Task PersistsRecordsAndSupportsFilteredVectorSearch()
    {
        CancellationToken cancellationToken = TestContext.Current.CancellationToken;
        var store = new TurboQuantFileSystemVectorStore(_directory, bitWidth: 4);
        TurboQuantCollection<string, Document> collection = store.GetCollection("documents", TestMappings.Document);
        await collection.EnsureCollectionExistsAsync(cancellationToken);
        await collection.UpsertAsync(
        [
            new Document("first", "manual", 1, Axis0),
            new Document("second", "guide", 2, Axis1),
            new Document("third", "guide", 3, Axis2)
        ], cancellationToken);

        Document? first = await collection.GetAsync("first", new RecordRetrievalOptions { IncludeVectors = true }, cancellationToken);
        Assert.NotNull(first);
        Assert.Equal(1, first.Rank);
        Assert.Equal(8, first.Embedding.Length);
        Document? withoutVector = await collection.GetAsync("first", cancellationToken: cancellationToken);
        Assert.NotNull(withoutVector);
        Assert.Equal("manual", withoutVector.Category);
        Assert.Equal(0, withoutVector.Embedding.Length);

        List<VectorSearchResult<Document>> results = await ToListAsync(collection.SearchAsync(
            new ReadOnlyMemory<float>(Axis1),
            top: 3,
            new VectorSearchOptions<Document>
            {
                Filter = document => document.Category == "guide",
                IncludeVectors = true
            },
            cancellationToken));

        Assert.Equal(["second", "third"], results.Select(result => result.Record.Id));
        Assert.All(results, result => Assert.Equal(8, result.Record.Embedding.Length));

        var reopenedStore = new TurboQuantFileSystemVectorStore(_directory, bitWidth: 4);
        TurboQuantCollection<string, Document> reopened = reopenedStore.GetCollection("documents", TestMappings.Document);
        Assert.True(await reopenedStore.CollectionExistsAsync("documents", cancellationToken));
        Assert.Equal(["documents"], await ToListAsync(reopenedStore.ListCollectionNamesAsync(cancellationToken)));
        Document? persisted = await reopened.GetAsync("third", new RecordRetrievalOptions { IncludeVectors = true }, cancellationToken);
        Assert.NotNull(persisted);
        Assert.Equal(3, persisted.Rank);

        List<Document> filtered = await ToListAsync(reopened.GetAsync(
            document => document.Category == "guide",
            top: 1,
            new FilteredRecordRetrievalOptions<Document>
            {
                OrderBy = orderBy => orderBy.Descending(document => document.Rank)
            },
            cancellationToken));

        Assert.Single(filtered);
        Assert.Equal("third", filtered[0].Id);
        Assert.Equal(0, filtered[0].Embedding.Length);
        Assert.True(await reopened.CollectionExistsAsync(cancellationToken));

        await reopened.DeleteAsync("second", cancellationToken);
        Assert.Null(await reopened.GetAsync("second", new RecordRetrievalOptions { IncludeVectors = true }, cancellationToken));
        await reopened.EnsureCollectionDeletedAsync(cancellationToken);
        Assert.False(await reopened.CollectionExistsAsync(cancellationToken));
    }

    [Fact]
    public async Task CosineSimilarityIgnoresVectorMagnitudeWithoutMutatingInputs()
    {
        CancellationToken cancellationToken = TestContext.Current.CancellationToken;
        var store = new TurboQuantFileSystemVectorStore(_directory);
        TurboQuantCollection<string, CosineDocument> collection = store.GetCollection("cosine", TestMappings.CosineDocument);
        float[] large = [100f, 100f, 0f, 0f, 0f, 0f, 0f, 0f];
        await collection.UpsertAsync(
        [
            new CosineDocument { Id = "aligned", Embedding = Axis0 },
            new CosineDocument { Id = "large-but-angled", Embedding = large }
        ], cancellationToken);

        List<VectorSearchResult<CosineDocument>> results = await ToListAsync(collection.SearchAsync(
            new ReadOnlyMemory<float>([3f, 0f, 0f, 0f, 0f, 0f, 0f, 0f]), top: 1, cancellationToken: cancellationToken));

        Assert.Equal("aligned", Assert.Single(results).Record.Id);
        Assert.Equal(100f, large[0]);
    }

    [Fact]
    public async Task SupportsDynamicRecordDefinitions()
    {
        CancellationToken cancellationToken = TestContext.Current.CancellationToken;
        var definition = new VectorStoreCollectionDefinition
        {
            Properties =
            [
                new VectorStoreKeyProperty("Id", typeof(string)),
                new VectorStoreDataProperty("Category", typeof(string)),
                new VectorStoreDataProperty("Rank", typeof(int?)),
                new VectorStoreVectorProperty("Embedding", typeof(ReadOnlyMemory<float>), 8)
            ]
        };
        var store = new TurboQuantFileSystemVectorStore(_directory);
        VectorStoreCollection<object, Dictionary<string, object?>> collection = store.GetDynamicCollection("dynamic", definition);
        await collection.UpsertAsync(new Dictionary<string, object?>
        {
            ["Id"] = "dynamic-1",
            ["Category"] = "notes",
            ["Rank"] = 7,
            ["Embedding"] = new ReadOnlyMemory<float>(Axis0)
        }, cancellationToken);

        Dictionary<string, object?>? record = await collection.GetAsync(
            "dynamic-1", new RecordRetrievalOptions { IncludeVectors = true }, cancellationToken);
        Assert.NotNull(record);
        Assert.Equal("notes", record["Category"]);
        Assert.Equal(7, record["Rank"]);
        Assert.IsType<ReadOnlyMemory<float>>(record["Embedding"]);

        Dictionary<string, object?>? withoutVector = await collection.GetAsync("dynamic-1", cancellationToken: cancellationToken);
        Assert.NotNull(withoutVector);
        Assert.Equal("notes", withoutVector["Category"]);
        Assert.Equal(7, withoutVector["Rank"]);
        Assert.False(withoutVector.ContainsKey("Embedding"));
    }

    [Fact]
    public async Task DynamicDefinitionsRequireJsonMetadataForCustomTypes()
    {
        var definition = new VectorStoreCollectionDefinition
        {
            Properties =
            [
                new VectorStoreKeyProperty("Id", typeof(string)),
                new VectorStoreDataProperty("Custom", typeof(CosineDocument)),
                new VectorStoreVectorProperty("Embedding", typeof(ReadOnlyMemory<float>), 8)
            ]
        };

        Assert.Throws<NotSupportedException>(() => new InMemoryVectorStore().GetDynamicCollection("dynamic", definition));

        var options = new JsonSerializerOptions { TypeInfoResolver = TestJsonContext.Default };
        var store = new TurboQuantFileSystemVectorStore(_directory, serializerOptions: options);
        VectorStoreCollection<object, Dictionary<string, object?>> collection = store.GetDynamicCollection("dynamic", definition);
        await collection.UpsertAsync(new Dictionary<string, object?>
        {
            ["Id"] = "x",
            ["Custom"] = new CosineDocument { Id = "nested" },
            ["Embedding"] = new ReadOnlyMemory<float>(Axis0)
        }, TestContext.Current.CancellationToken);

        Dictionary<string, object?>? record = await collection.GetAsync("x", cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal("nested", Assert.IsType<CosineDocument>(record?["Custom"]).Id);
    }

    [Fact]
    public async Task RegisteredMappingsServeTheStandardTypedCollectionApi()
    {
        CancellationToken cancellationToken = TestContext.Current.CancellationToken;
        VectorStore store = new TurboQuantFileSystemVectorStore(_directory).RegisterRecordMapping(TestMappings.Document);
        VectorStoreCollection<string, Document> collection = store.GetCollection<string, Document>("documents");
        await collection.UpsertAsync(new Document("a", "guide", 1, Axis0), cancellationToken);

        Assert.Equal("guide", (await collection.GetAsync("a", cancellationToken: cancellationToken))?.Category);
        InvalidOperationException error = Assert.Throws<InvalidOperationException>(
            () => store.GetCollection<string, CosineDocument>("cosine"));
        Assert.Contains("RegisterRecordMapping", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void MappingsAreValidatedAgainstJsonMetadata()
    {
        var mapping = new TurboQuantRecordMapping<string, Document>
        {
            Definition = new VectorStoreCollectionDefinition
            {
                Properties =
                [
                    new VectorStoreKeyProperty("Id", typeof(string)),
                    new VectorStoreDataProperty("Category", typeof(string)),
                    new VectorStoreVectorProperty("Embedding", typeof(ReadOnlyMemory<float>), 8)
                ]
            },
            RecordTypeInfo = TestJsonContext.Default.Document,
            KeyTypeInfo = TestJsonContext.Default.String,
            GetKey = static document => document.Id,
            GetVector = static document => document.Embedding
        };

        InvalidOperationException error = Assert.Throws<InvalidOperationException>(
            () => new InMemoryVectorStore().GetCollection("documents", mapping));
        Assert.Contains("StorageName", error.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(7)]
    [InlineData(16392)]
    public void UnsupportedDimensionsAreRejectedWhenCreatingCollections(int dimension)
    {
        var mapping = new TurboQuantRecordMapping<string, Document>
        {
            Definition = new VectorStoreCollectionDefinition
            {
                Properties =
                [
                    new VectorStoreKeyProperty("Id", typeof(string)),
                    new VectorStoreVectorProperty("Embedding", typeof(ReadOnlyMemory<float>), dimension)
                ]
            },
            RecordTypeInfo = TestJsonContext.Default.Document,
            KeyTypeInfo = TestJsonContext.Default.String,
            GetKey = static document => document.Id,
            GetVector = static document => document.Embedding
        };

        NotSupportedException error = Assert.Throws<NotSupportedException>(
            () => new InMemoryVectorStore().GetCollection("documents", mapping));
        Assert.Contains("dimensions", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task GeneratesGuidKeys()
    {
        CancellationToken cancellationToken = TestContext.Current.CancellationToken;
        var store = new InMemoryVectorStore();
        TurboQuantCollection<Guid, GuidDocument> collection = store.GetCollection("guids", TestMappings.GuidDocument);
        var document = new GuidDocument { Embedding = Axis0 };
        await collection.UpsertAsync(document, cancellationToken);

        Assert.NotEqual(Guid.Empty, document.Id);
        Assert.NotNull(await collection.GetAsync(document.Id, cancellationToken: cancellationToken));
    }

    [Fact]
    public async Task CustomPersistenceBackendsReuseTheCollectionImplementation()
    {
        CancellationToken cancellationToken = TestContext.Current.CancellationToken;
        var store = new InMemoryVectorStore(bitWidth: 3);
        TurboQuantCollection<string, Document> collection = store.GetCollection("documents", TestMappings.Document);
        await collection.UpsertAsync([new Document("a", "x", 1, Axis0), new Document("b", "y", 2, Axis1)], cancellationToken);

        Assert.Equal(["documents"], await ToListAsync(store.ListCollectionNamesAsync(cancellationToken)));
        Assert.True(await store.CollectionExistsAsync("documents", cancellationToken));
        List<VectorSearchResult<Document>> results = await ToListAsync(collection.SearchAsync(
            new ReadOnlyMemory<float>(Axis1), top: 1, cancellationToken: cancellationToken));
        Assert.Equal("b", Assert.Single(results).Record.Id);

        store.FailWrites = true;
        await Assert.ThrowsAsync<IOException>(() => collection.UpsertAsync(new Document("c", "z", 3, Axis2), cancellationToken));
        store.FailWrites = false;
        Assert.Null(await collection.GetAsync("c", cancellationToken: cancellationToken));
        Assert.NotNull(await collection.GetAsync("a", cancellationToken: cancellationToken));

        await store.EnsureCollectionDeletedAsync("documents", cancellationToken);
        Assert.Empty(store.Images);
        Assert.False(await collection.CollectionExistsAsync(cancellationToken));
    }

    [Fact]
    public async Task CancellationBeforeBackendCommitPreservesTheExistingImage()
    {
        CancellationToken cancellationToken = TestContext.Current.CancellationToken;
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var store = new InMemoryVectorStore();
        TurboQuantCollection<string, Document> collection = store.GetCollection("documents", TestMappings.Document);
        await collection.UpsertAsync(new Document("a", "original", 1, Axis0), cancellationToken);
        byte[] committed = store.Images["documents"].ToArray();

        store.BeforeCommit = cancellation.Cancel;
        OperationCanceledException error = await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => collection.UpsertAsync(new Document("b", "cancelled", 2, Axis1), cancellation.Token));
        store.BeforeCommit = null;

        Assert.Equal(cancellation.Token, error.CancellationToken);
        Assert.Equal(committed, store.Images["documents"]);
        Assert.Equal("original", (await collection.GetAsync("a", cancellationToken: cancellationToken))?.Category);
        Assert.Null(await collection.GetAsync("b", cancellationToken: cancellationToken));
    }

    [Fact]
    public async Task CancelledFileWritesPreserveTheCommittedImage()
    {
        CancellationToken cancellationToken = TestContext.Current.CancellationToken;
        var store = new TurboQuantFileSystemVectorStore(_directory);
        TurboQuantCollection<string, Document> collection = store.GetCollection("documents", TestMappings.Document);
        await collection.UpsertAsync(new Document("a", "original", 1, Axis0), cancellationToken);
        string imagePath = Assert.Single(Directory.GetFiles(_directory, "*.tqv"));
        byte[] committed = await File.ReadAllBytesAsync(imagePath, cancellationToken);
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        cancellation.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => collection.UpsertAsync(new Document("b", "cancelled", 2, Axis1), cancellation.Token));

        Assert.Equal(committed, await File.ReadAllBytesAsync(imagePath, cancellationToken));
        Assert.Empty(Directory.GetFiles(_directory, "*.tmp"));
        Assert.Null(await collection.GetAsync("b", cancellationToken: cancellationToken));
    }

    [Fact]
    public async Task FileStoreHashesArbitraryNamesAndListsThemOrdinally()
    {
        CancellationToken cancellationToken = TestContext.Current.CancellationToken;
        string[] names = ["../outside", "Alpha", "alpha", "docs/nested", "n\u00e4me"];
        var store = new TurboQuantFileSystemVectorStore(_directory);
        foreach (string name in names)
        {
            await store.GetCollection(name, TestMappings.Document).EnsureCollectionExistsAsync(cancellationToken);
        }

        Assert.Equal(names.Order(StringComparer.Ordinal), await ToListAsync(store.ListCollectionNamesAsync(cancellationToken)));
        IEnumerable<string> expectedFiles = names
            .Select(name => Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(name))) + ".tqv")
            .Order(StringComparer.Ordinal);
        IEnumerable<string?> actualFiles = Directory.GetFiles(_directory).Select(Path.GetFileName).Order(StringComparer.Ordinal);
        Assert.Equal(expectedFiles, actualFiles);
    }

    [Fact]
    public async Task FileStoresCoordinateWritesAcrossInstances()
    {
        CancellationToken cancellationToken = TestContext.Current.CancellationToken;
        TurboQuantCollection<string, Document>[] collections =
        [
            new TurboQuantFileSystemVectorStore(_directory).GetCollection("shared", TestMappings.Document),
            new TurboQuantFileSystemVectorStore(_directory).GetCollection("shared", TestMappings.Document)
        ];

        await Task.WhenAll(Enumerable.Range(0, 20).Select(i =>
            collections[i % 2].UpsertAsync(new Document($"doc-{i}", "c", i, Axis0), cancellationToken)));

        List<Document> all = await ToListAsync(collections[0].GetAsync(_ => true, top: 100, cancellationToken: cancellationToken));
        Assert.Equal(20, all.Count);
    }

    [Fact]
    public async Task SchemaFingerprintsMatchTheReflectionBasedModel()
    {
        CancellationToken cancellationToken = TestContext.Current.CancellationToken;
        var store = new InMemoryVectorStore();
        await store.GetCollection("documents", TestMappings.Document).EnsureCollectionExistsAsync(cancellationToken);
        await store.GetCollection("cosine", TestMappings.CosineDocument).EnsureCollectionExistsAsync(cancellationToken);
        await store.GetCollection("guids", TestMappings.GuidDocument).EnsureCollectionExistsAsync(cancellationToken);

        Assert.Equal(LegacyFingerprint(typeof(Document), typeof(string)), StoredSchema(store, "documents"));
        Assert.Equal(LegacyFingerprint(typeof(CosineDocument), typeof(string)), StoredSchema(store, "cosine"));
        Assert.Equal(LegacyFingerprint(typeof(GuidDocument), typeof(Guid)), StoredSchema(store, "guids"));
    }

    [Fact]
    public async Task ReadsCollectionImagesWrittenByEarlierVersions()
    {
        CancellationToken cancellationToken = TestContext.Current.CancellationToken;
        IdMapIndex index = IdMapIndex.NewLazy(4);
        index.AddWithIds2D([.. Axis0, .. Axis1], 8, [1UL, 2UL]);
        var legacyImage = new JsonObject
        {
            ["Name"] = "legacy",
            ["Schema"] = LegacyFingerprint(typeof(Document), typeof(string)),
            ["Index"] = Convert.ToBase64String(index.ToBytes()),
            ["KeyToId"] = new JsonObject { ["\"a\""] = 1, ["\"b\""] = 2 },
            ["Records"] = new JsonObject
            {
                ["1"] = JsonSerializer.Serialize(new Document("a", "first", 1, Axis0), TestJsonContext.Default.Document),
                ["2"] = JsonSerializer.Serialize(new Document("b", "second", 2, Axis1), TestJsonContext.Default.Document)
            },
            ["NextId"] = 3
        };
        var store = new InMemoryVectorStore();
        store.Images["legacy"] = JsonSerializer.SerializeToUtf8Bytes<JsonNode>(legacyImage);

        TurboQuantCollection<string, Document> collection = store.GetCollection("legacy", TestMappings.Document);
        Assert.Equal("second", (await collection.GetAsync("b", cancellationToken: cancellationToken))?.Category);
        List<VectorSearchResult<Document>> results = await ToListAsync(collection.SearchAsync(
            new ReadOnlyMemory<float>(Axis1), top: 1, cancellationToken: cancellationToken));
        Assert.Equal("b", Assert.Single(results).Record.Id);

        await collection.UpsertAsync(new Document("c", "third", 3, Axis2), cancellationToken);
        JsonObject rewritten = JsonNode.Parse(store.Images["legacy"])!.AsObject();
        Assert.Equal(["Name", "Schema", "Index", "KeyToId", "Records", "NextId"], rewritten.Select(property => property.Key));
        Assert.Equal(4UL, rewritten["NextId"]!.GetValue<ulong>());
    }

    [Fact]
    public async Task ReadsEarlierImagesWrittenWithAJsonNamingPolicy()
    {
        CancellationToken cancellationToken = TestContext.Current.CancellationToken;
        var legacyOptions = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase };
        var context = new TestJsonContext(new JsonSerializerOptions(legacyOptions));
        var mapping = new TurboQuantRecordMapping<string, Document>
        {
            Definition = TestMappings.Document.Definition,
            RecordTypeInfo = context.Document,
            KeyTypeInfo = context.String,
            GetKey = static document => document.Id,
            GetVector = static document => document.Embedding
        };
        var document = new Document("legacy-id", "original", 1, Axis0);
        IdMapIndex index = IdMapIndex.NewLazy(4);
        index.AddWithIds2D(Axis0, 8, [1UL]);
        var legacyImage = new JsonObject
        {
            ["Name"] = "legacy-camel",
            ["Schema"] = LegacyFingerprint(typeof(Document), typeof(string), legacyOptions),
            ["Index"] = Convert.ToBase64String(index.ToBytes()),
            ["KeyToId"] = new JsonObject { [JsonSerializer.Serialize(document.Id, legacyOptions)] = 1 },
            ["Records"] = new JsonObject { ["1"] = JsonSerializer.Serialize(document, legacyOptions) },
            ["NextId"] = 2
        };
        var store = new InMemoryVectorStore();
        store.Images["legacy-camel"] = JsonSerializer.SerializeToUtf8Bytes<JsonNode>(legacyImage);

        TurboQuantCollection<string, Document> collection = store.GetCollection("legacy-camel", mapping);
        Document? restored = await collection.GetAsync(
            "legacy-id", new RecordRetrievalOptions { IncludeVectors = true }, cancellationToken);
        Assert.NotNull(restored);
        Assert.Equal("original", restored.Category);
        Assert.Equal(Axis0, restored.Embedding.ToArray());

        await collection.UpsertAsync(new Document("next", "new", 2, Axis1), cancellationToken);
        Assert.Equal("original", (await collection.GetAsync("legacy-id", cancellationToken: cancellationToken))?.Category);
    }

    private static string StoredSchema(InMemoryVectorStore store, string name)
        => JsonNode.Parse(store.Images[name])!["Schema"]!.GetValue<string>();

    // Reproduces the fingerprint written by the reflection-based provider releases.
    private static string LegacyFingerprint(Type recordType, Type keyType, JsonSerializerOptions? options = null)
    {
#pragma warning disable IL2026, IL3050
        CollectionModel model = new LegacyModelBuilder().Build(
            recordType, keyType, definition: null, defaultEmbeddingGenerator: null,
            options ?? new JsonSerializerOptions(JsonSerializerDefaults.General));
#pragma warning restore IL2026, IL3050
        var entries = new List<string>();
        foreach (PropertyModel property in model.Properties)
        {
            string kind = property switch
            {
                KeyPropertyModel => "key",
                VectorPropertyModel vector => $"vector:{vector.Dimensions}:{vector.DistanceFunction}",
                _ => "data"
            };
            entries.Add($"{kind}|{property.ModelName}|{property.StorageName}|{property.Type.AssemblyQualifiedName}");
        }

        entries.Sort(StringComparer.Ordinal);
        return string.Join('\n', entries);
    }

    private static async Task<List<T>> ToListAsync<T>(IAsyncEnumerable<T> source)
    {
        var list = new List<T>();
        await foreach (T item in source)
        {
            list.Add(item);
        }

        return list;
    }

    private sealed class LegacyModelBuilder() : CollectionJsonModelBuilder(new CollectionModelBuildingOptions
    {
        SupportsMultipleVectors = false,
        RequiresAtLeastOneVector = true,
        UsesExternalSerializer = true
    })
    {
        protected override bool IsDataPropertyTypeValid(Type type, out string errorMessage)
        {
            errorMessage = string.Empty;
            return true;
        }

        protected override bool IsVectorPropertyTypeValid(Type type, out string errorMessage)
        {
            errorMessage = string.Empty;
            return true;
        }
    }
}
