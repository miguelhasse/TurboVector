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

    [Theory]
    [InlineData(2, false)]
    [InlineData(2, true)]
    [InlineData(3, false)]
    [InlineData(3, true)]
    [InlineData(4, false)]
    [InlineData(4, true)]
    public async Task ProviderSearchPreservesCoreScoresAndOrdering(int bits, bool filtered)
    {
        const int dim = 8;
        const int count = 129;
        var cancellationToken = TestContext.Current.CancellationToken;
        var vectors = new float[count * dim];
        var state = 0xc0ffeeUL;
        for (var i = 0; i < vectors.Length; i++)
        {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            vectors[i] = ((int)((state >> 32) & 0xffff) - 32768) / 32768f;
        }

        var records = new Document[count];
        for (var i = 0; i < count; i++)
        {
            records[i] = new Document(i.ToString(), "parity", i, vectors.AsMemory(i * dim, dim));
        }
        var collection = new InMemoryVectorStore(bits).GetCollection("parity", TestMappings.Document);
        await collection.UpsertAsync(records, cancellationToken);
        var query = new float[] { 0.5f, -0.75f, 0.25f, 1f, -1f, 0.125f, -0.5f, 0.375f };
        var options = new VectorSearchOptions<Document>
        {
            Filter = filtered ? document => document.Rank % 2 == 0 : null
        };
        var actual = await ToListAsync(collection.SearchAsync(query, 5, options, cancellationToken));

        var index = new TurboQuantIndex(dim, bits);
        index.Add(vectors);
        var mask = filtered ? Enumerable.Range(0, count).Select(static i => i % 2 == 0).ToArray() : [];
        var expected = index.SearchWithMask(query, 5, mask);
        Assert.Equal(expected.Indices.Select(static id => id.ToString()), actual.Select(static result => result.Record.Id));
        Assert.Equal(expected.Scores.Select(static score => (double)score),
            actual.Select(static result => Assert.IsType<double>(result.Score)));
    }

    [Theory]
    [InlineData(false, false, false)]
    [InlineData(false, false, true)]
    [InlineData(false, true, false)]
    [InlineData(false, true, true)]
    [InlineData(true, false, false)]
    [InlineData(true, false, true)]
    [InlineData(true, true, false)]
    [InlineData(true, true, true)]
    public async Task FilteredOrderingUsesVectorsBeforeProjection(bool includeVectors, bool ascendingVector, bool descendingRank)
    {
        var cancellationToken = TestContext.Current.CancellationToken;
        var store = new InMemoryVectorStore();
        var collection = store.GetCollection("documents", TestMappings.Document);
        await collection.UpsertAsync(
        [
            new Document("first", "guide", 1, Axis0),
            new Document("second", "guide", 3, Axis1),
            new Document("third", "guide", 2, Axis2)
        ], cancellationToken);

        var result = await ToListAsync(collection.GetAsync(
            document => document.Category == "guide", 2,
            new FilteredRecordRetrievalOptions<Document>
            {
                IncludeVectors = includeVectors,
                Skip = 1,
                OrderBy = orderBy =>
                {
                    var primary = ascendingVector
                        ? orderBy.Ascending(document => document.Embedding.ToArray()[0])
                        : orderBy.Descending(document => document.Embedding.ToArray()[0]);
                    return descendingRank
                        ? primary.Descending(document => document.Rank)
                        : primary.Ascending(document => document.Rank);
                }
            }, cancellationToken));

        var expectedIds = (ascendingVector, descendingRank) switch
        {
            (false, false) => new[] { "third", "second" },
            (false, true) => ["second", "third"],
            (true, false) => ["second", "first"],
            (true, true) => ["third", "first"]
        };
        Assert.Equal(expectedIds, result.Select(document => document.Id));
        Assert.All(result, document => Assert.Equal(includeVectors ? 8 : 0, document.Embedding.Length));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public async Task ScoreThresholdStillHonorsTopAndSkip(int skip)
    {
        CancellationToken cancellationToken = TestContext.Current.CancellationToken;
        var store = new InMemoryVectorStore();
        TurboQuantCollection<string, Document> collection = store.GetCollection("documents", TestMappings.Document);
        await collection.UpsertAsync(
        [
            new Document("first", "guide", 1, Axis0),
            new Document("second", "guide", 2, Axis1),
            new Document("third", "guide", 3, Axis2)
        ], cancellationToken);

        List<VectorSearchResult<Document>> expected = await ToListAsync(collection.SearchAsync(
            new ReadOnlyMemory<float>(Axis0), 1, new VectorSearchOptions<Document> { Skip = skip }, cancellationToken));
        List<VectorSearchResult<Document>> actual = await ToListAsync(collection.SearchAsync(
            new ReadOnlyMemory<float>(Axis0), 1,
            new VectorSearchOptions<Document> { Skip = skip, ScoreThreshold = -2 }, cancellationToken));

        Assert.Single(expected);
        Assert.Equal(expected.Select(result => result.Record.Id), actual.Select(result => result.Record.Id));
    }

    [Theory]
    [InlineData(int.MaxValue, 1, 2)]
    [InlineData(1, int.MaxValue, 0)]
    public async Task SearchBoundsLargeTopAndSkipToCollectionSize(int top, int skip, int expectedCount)
    {
        CancellationToken cancellationToken = TestContext.Current.CancellationToken;
        var store = new InMemoryVectorStore();
        TurboQuantCollection<string, Document> collection = store.GetCollection("documents", TestMappings.Document);
        await collection.UpsertAsync(
        [
            new Document("first", "guide", 1, Axis0),
            new Document("second", "guide", 2, Axis1),
            new Document("third", "guide", 3, Axis2)
        ], cancellationToken);

        List<VectorSearchResult<Document>> results = await ToListAsync(collection.SearchAsync(
            new ReadOnlyMemory<float>(Axis0), top, new VectorSearchOptions<Document> { Skip = skip }, cancellationToken));

        Assert.Equal(expectedCount, results.Count);
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

    [Theory]
    [InlineData(1e-11f, false)]
    [InlineData(1e-20f, false)]
    [InlineData(1e-30f, false)]
    [InlineData(float.Epsilon, false)]
    [InlineData(1e15f, false)]
    [InlineData(9e15f, false)]
    [InlineData(1e-11f, true)]
    [InlineData(1e-20f, true)]
    [InlineData(1e-30f, true)]
    [InlineData(float.Epsilon, true)]
    [InlineData(1e15f, true)]
    [InlineData(9e15f, true)]
    public async Task CosineSearchPreservesDirectionOfExtremeVectors(float magnitude, bool useDistance)
    {
        var cancellationToken = TestContext.Current.CancellationToken;
        var store = new InMemoryVectorStore();
        var mapping = useDistance
            ? new TurboQuantRecordMapping<string, CosineDocument>
            {
                Definition = new VectorStoreCollectionDefinition
                {
                    Properties =
                    [
                        new VectorStoreKeyProperty("Id", typeof(string)),
                        new VectorStoreVectorProperty("Embedding", typeof(ReadOnlyMemory<float>), 8)
                        {
                            DistanceFunction = DistanceFunction.CosineDistance
                        }
                    ]
                },
                RecordTypeInfo = TestJsonContext.Default.CosineDocument,
                KeyTypeInfo = TestJsonContext.Default.String,
                GetKey = static document => document.Id,
                GetVector = static document => document.Embedding
            }
            : TestMappings.CosineDocument;
        var baseline = store.GetCollection("baseline", mapping);
        var scaled = store.GetCollection("scaled", mapping);
        var direction = new[] { 1f, 1f, 0f, 0f, 0f, 0f, 0f, 0f };
        var orthogonal = new[] { -1f, 1f, 0f, 0f, 0f, 0f, 0f, 0f };
        var tiny = new[] { magnitude, magnitude, 0f, 0f, 0f, 0f, 0f, 0f };
        await baseline.UpsertAsync(
        [
            new CosineDocument { Id = "orthogonal", Embedding = orthogonal },
            new CosineDocument { Id = "aligned", Embedding = direction }
        ], cancellationToken);
        await scaled.UpsertAsync(
        [
            new CosineDocument { Id = "orthogonal", Embedding = orthogonal },
            new CosineDocument { Id = "aligned", Embedding = tiny }
        ], cancellationToken);

        var expected = await ToListAsync(baseline.SearchAsync(direction, 2, cancellationToken: cancellationToken));
        var tinyQuery = await ToListAsync(baseline.SearchAsync(tiny, 2, cancellationToken: cancellationToken));
        var tinyStoredVector = await ToListAsync(scaled.SearchAsync(direction, 2, cancellationToken: cancellationToken));

        Assert.Equal("aligned", expected[0].Record.Id);
        Assert.Equal(expected.Select(result => result.Record.Id), tinyQuery.Select(result => result.Record.Id));
        Assert.Equal(expected.Select(result => result.Record.Id), tinyStoredVector.Select(result => result.Record.Id));
        var expectedScore = Assert.IsType<double>(expected[0].Score);
        var queryScore = Assert.IsType<double>(tinyQuery[0].Score);
        var storedScore = Assert.IsType<double>(tinyStoredVector[0].Score);
        Assert.True(useDistance ? queryScore < 0.2 : queryScore > 0.8);
        Assert.True(useDistance ? storedScore < 0.2 : storedScore > 0.8);
        Assert.InRange(Math.Abs(expectedScore - queryScore), 0, 1e-5);
        Assert.InRange(Math.Abs(expectedScore - storedScore), 0, 1e-5);
        Assert.Equal(magnitude, tiny[0]);
        Assert.Equal(magnitude, tiny[1]);
    }

    [Fact]
    public async Task CosineSearchAcceptsZeroVectors()
    {
        var cancellationToken = TestContext.Current.CancellationToken;
        var store = new InMemoryVectorStore();
        var collection = store.GetCollection("zero", TestMappings.CosineDocument);
        var zero = new float[8];
        await collection.UpsertAsync(new CosineDocument { Id = "zero", Embedding = zero }, cancellationToken);

        var result = Assert.Single(await ToListAsync(collection.SearchAsync(zero, 1, cancellationToken: cancellationToken)));

        Assert.Equal(0d, result.Score);
        Assert.All(zero, value => Assert.Equal(0f, value));
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
    public async Task DynamicRecordsRespectDictionaryKeyPolicies()
    {
        CancellationToken cancellationToken = TestContext.Current.CancellationToken;
        var definition = new VectorStoreCollectionDefinition
        {
            Properties =
            [
                new VectorStoreKeyProperty("Id", typeof(string)),
                new VectorStoreDataProperty("Rank", typeof(int)),
                new VectorStoreVectorProperty("Embedding", typeof(ReadOnlyMemory<float>), 8)
            ]
        };
        var options = new JsonSerializerOptions { DictionaryKeyPolicy = JsonNamingPolicy.CamelCase };
        var store = new TurboQuantFileSystemVectorStore(_directory, serializerOptions: options);
        VectorStoreCollection<object, Dictionary<string, object?>> collection = store.GetDynamicCollection("dynamic", definition);
        await collection.UpsertAsync(new Dictionary<string, object?>
        {
            ["Id"] = "dynamic-1",
            ["Rank"] = 7,
            ["Embedding"] = new ReadOnlyMemory<float>(Axis0)
        }, cancellationToken);

        Dictionary<string, object?>? full = await collection.GetAsync(
            "dynamic-1", new RecordRetrievalOptions { IncludeVectors = true }, cancellationToken);
        Assert.NotNull(full);
        Assert.Equal("dynamic-1", full["Id"]);
        Assert.Equal(7, full["Rank"]);
        Assert.IsType<ReadOnlyMemory<float>>(full["Embedding"]);

        Dictionary<string, object?>? withoutVector = await collection.GetAsync("dynamic-1", cancellationToken: cancellationToken);
        Assert.NotNull(withoutVector);
        Assert.Equal(7, withoutVector["Rank"]);
        Assert.False(withoutVector.ContainsKey("Embedding"));
        Assert.False(withoutVector.ContainsKey("embedding"));

        var ordered = await ToListAsync(collection.GetAsync(
            record => (int)record["Rank"]! == 7, 1,
            new FilteredRecordRetrievalOptions<Dictionary<string, object?>>
            {
                OrderBy = orderBy => orderBy.Descending(
                    record => ((ReadOnlyMemory<float>)record["Embedding"]!).ToArray()[0])
            }, cancellationToken));
        Assert.Equal("dynamic-1", Assert.Single(ordered)["Id"]);
        Assert.False(ordered[0].ContainsKey("Embedding"));

        List<VectorSearchResult<Dictionary<string, object?>>> results = await ToListAsync(collection.SearchAsync(
            new ReadOnlyMemory<float>(Axis0), 1,
            new VectorSearchOptions<Dictionary<string, object?>> { Filter = record => (int)record["Rank"]! == 7 },
            cancellationToken));
        Assert.Equal("dynamic-1", Assert.Single(results).Record["Id"]);
    }

    [Fact]
    public async Task DynamicRecordsRejectDictionaryKeyPolicyCollisions()
    {
        CancellationToken cancellationToken = TestContext.Current.CancellationToken;
        var definition = new VectorStoreCollectionDefinition
        {
            Properties =
            [
                new VectorStoreKeyProperty("Id", typeof(string)),
                new VectorStoreVectorProperty("Embedding", typeof(ReadOnlyMemory<float>), 8)
            ]
        };
        var options = new JsonSerializerOptions { DictionaryKeyPolicy = JsonNamingPolicy.CamelCase };
        var store = new TurboQuantFileSystemVectorStore(_directory, serializerOptions: options);
        VectorStoreCollection<object, Dictionary<string, object?>> collection = store.GetDynamicCollection("dynamic", definition);

        await Assert.ThrowsAsync<ArgumentException>(() => collection.UpsertAsync(new Dictionary<string, object?>
        {
            ["Id"] = "dynamic-1",
            ["id"] = "conflicting",
            ["Embedding"] = new ReadOnlyMemory<float>(Axis0)
        }, cancellationToken));
        Assert.False(await collection.CollectionExistsAsync(cancellationToken));
    }

    [Fact]
    public void DynamicDefinitionsRejectDictionaryKeyPolicyCollisions()
    {
        var definition = new VectorStoreCollectionDefinition
        {
            Properties =
            [
                new VectorStoreKeyProperty("Id", typeof(string)),
                new VectorStoreDataProperty("id", typeof(string)),
                new VectorStoreVectorProperty("Embedding", typeof(ReadOnlyMemory<float>), 8)
            ]
        };
        var options = new JsonSerializerOptions { DictionaryKeyPolicy = JsonNamingPolicy.CamelCase };

        InvalidOperationException error = Assert.Throws<InvalidOperationException>(
            () => new TurboQuantFileSystemVectorStore(_directory, serializerOptions: options).GetDynamicCollection("dynamic", definition));
        Assert.Contains("multiple properties", error.Message, StringComparison.Ordinal);
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

    [Fact]
    public async Task RequiredVectorPropertiesCanBeExcludedFromRecords()
    {
        CancellationToken cancellationToken = TestContext.Current.CancellationToken;
        var mapping = new TurboQuantRecordMapping<string, RequiredDocument>
        {
            Definition = new VectorStoreCollectionDefinition
            {
                Properties =
                [
                    new VectorStoreKeyProperty("Id", typeof(string)),
                    new VectorStoreVectorProperty("Embedding", typeof(ReadOnlyMemory<float>), 8)
                ]
            },
            RecordTypeInfo = TestJsonContext.Default.RequiredDocument,
            KeyTypeInfo = TestJsonContext.Default.String,
            GetKey = static document => document.Id,
            GetVector = static document => document.Embedding
        };
        var store = new InMemoryVectorStore();
        TurboQuantCollection<string, RequiredDocument> collection = store.GetCollection("required", mapping);
        await collection.UpsertAsync(new RequiredDocument { Id = "a", Embedding = Axis0 }, cancellationToken);

        RequiredDocument? result = await collection.GetAsync("a", cancellationToken: cancellationToken);
        Assert.NotNull(result);
        Assert.Equal("a", result.Id);
        Assert.True(result.Embedding.IsEmpty);

        List<VectorSearchResult<RequiredDocument>> results = await ToListAsync(collection.SearchAsync(
            new ReadOnlyMemory<float>(Axis0), 1, cancellationToken: cancellationToken));
        Assert.True(Assert.Single(results).Record.Embedding.IsEmpty);
    }

    [Fact]
    public async Task RequiredVectorsWithCustomJsonConvertersCanBeExcluded()
    {
        CancellationToken cancellationToken = TestContext.Current.CancellationToken;
        var mapping = new TurboQuantRecordMapping<string, RequiredConvertedDocument>
        {
            Definition = new VectorStoreCollectionDefinition
            {
                Properties =
                [
                    new VectorStoreKeyProperty("Id", typeof(string)),
                    new VectorStoreVectorProperty("Embedding", typeof(ReadOnlyMemory<float>), 8)
                ]
            },
            RecordTypeInfo = TestJsonContext.Default.RequiredConvertedDocument,
            KeyTypeInfo = TestJsonContext.Default.String,
            GetKey = static document => document.Id,
            GetVector = static document => document.Embedding
        };
        var store = new InMemoryVectorStore();
        TurboQuantCollection<string, RequiredConvertedDocument> collection = store.GetCollection("required", mapping);
        await collection.UpsertAsync(new RequiredConvertedDocument { Id = "a", Embedding = Axis0 }, cancellationToken);

        RequiredConvertedDocument? full = await collection.GetAsync(
            "a", new RecordRetrievalOptions { IncludeVectors = true }, cancellationToken);
        Assert.NotNull(full);
        Assert.Equal(Axis0, full.Embedding.ToArray());
        RequiredConvertedDocument? withoutVector = await collection.GetAsync("a", cancellationToken: cancellationToken);
        Assert.NotNull(withoutVector);
        Assert.True(withoutVector.Embedding.IsEmpty);
    }

    [Fact]
    public async Task RequiredVectorsWithGlobalJsonConvertersCanBeExcluded()
    {
        CancellationToken cancellationToken = TestContext.Current.CancellationToken;
        var options = new JsonSerializerOptions();
        options.Converters.Add(new StringVectorConverter());
        var context = new TestJsonContext(options);
        var mapping = new TurboQuantRecordMapping<string, RequiredDocument>
        {
            Definition = new VectorStoreCollectionDefinition
            {
                Properties =
                [
                    new VectorStoreKeyProperty("Id", typeof(string)),
                    new VectorStoreVectorProperty("Embedding", typeof(ReadOnlyMemory<float>), 8)
                ]
            },
            RecordTypeInfo = context.RequiredDocument,
            KeyTypeInfo = context.String,
            GetKey = static document => document.Id,
            GetVector = static document => document.Embedding
        };
        var store = new InMemoryVectorStore();
        TurboQuantCollection<string, RequiredDocument> collection = store.GetCollection("required", mapping);
        await collection.UpsertAsync(new RequiredDocument { Id = "a", Embedding = Axis0 }, cancellationToken);

        RequiredDocument? withoutVector = await collection.GetAsync("a", cancellationToken: cancellationToken);
        Assert.NotNull(withoutVector);
        Assert.True(withoutVector.Embedding.IsEmpty);
    }

    [Fact]
    public async Task RequiredCollectionVectorsCanBeExcluded()
    {
        CancellationToken cancellationToken = TestContext.Current.CancellationToken;
        var mapping = new TurboQuantRecordMapping<string, RequiredListDocument>
        {
            Definition = new VectorStoreCollectionDefinition
            {
                Properties =
                [
                    new VectorStoreKeyProperty("Id", typeof(string)),
                    new VectorStoreVectorProperty("Embedding", typeof(List<float>), 8)
                ]
            },
            RecordTypeInfo = TestJsonContext.Default.RequiredListDocument,
            KeyTypeInfo = TestJsonContext.Default.String,
            GetKey = static document => document.Id,
            GetVector = static document => document.Embedding.ToArray()
        };
        var store = new InMemoryVectorStore();
        TurboQuantCollection<string, RequiredListDocument> collection = store.GetCollection("required", mapping);
        await collection.UpsertAsync(new RequiredListDocument { Id = "a", Embedding = [.. Axis0] }, cancellationToken);

        RequiredListDocument? withoutVector = await collection.GetAsync("a", cancellationToken: cancellationToken);
        Assert.NotNull(withoutVector);
        Assert.Empty(withoutVector.Embedding);
    }

    [Fact]
    public async Task RequiredImmutableVectorsCanBeExcluded()
    {
        CancellationToken cancellationToken = TestContext.Current.CancellationToken;
        var mapping = new TurboQuantRecordMapping<string, RequiredImmutableDocument>
        {
            Definition = new VectorStoreCollectionDefinition
            {
                Properties =
                [
                    new VectorStoreKeyProperty("Id", typeof(string)),
                    new VectorStoreVectorProperty("Embedding", typeof(System.Collections.Immutable.ImmutableArray<float>), 8)
                ]
            },
            RecordTypeInfo = TestJsonContext.Default.RequiredImmutableDocument,
            KeyTypeInfo = TestJsonContext.Default.String,
            GetKey = static document => document.Id,
            GetVector = static document => document.Embedding.ToArray()
        };
        var store = new InMemoryVectorStore();
        TurboQuantCollection<string, RequiredImmutableDocument> collection = store.GetCollection("required", mapping);
        await collection.UpsertAsync(new RequiredImmutableDocument { Id = "a", Embedding = [.. Axis0] }, cancellationToken);

        RequiredImmutableDocument? withoutVector = await collection.GetAsync("a", cancellationToken: cancellationToken);
        Assert.NotNull(withoutVector);
        Assert.Empty(withoutVector.Embedding);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task RequiredNullableVectorsCanBeExcluded(bool useFactory)
    {
        CancellationToken cancellationToken = TestContext.Current.CancellationToken;
        var mapping = new TurboQuantRecordMapping<string, RequiredNullableDocument>
        {
            Definition = new VectorStoreCollectionDefinition
            {
                Properties =
                [
                    new VectorStoreKeyProperty("Id", typeof(string)),
                    new VectorStoreVectorProperty("Embedding", typeof(ReadOnlyMemory<float>?), 8)
                ]
            },
            RecordTypeInfo = TestJsonContext.Default.RequiredNullableDocument,
            KeyTypeInfo = TestJsonContext.Default.String,
            GetKey = static document => document.Id,
            GetVector = static document => document.Embedding.GetValueOrDefault(),
            CreateEmptyVector = useFactory ? static () => ReadOnlyMemory<float>.Empty : null
        };
        var store = new InMemoryVectorStore();
        TurboQuantCollection<string, RequiredNullableDocument> collection = store.GetCollection("nullable", mapping);
        await collection.UpsertAsync(new RequiredNullableDocument { Id = "a", Embedding = Axis0 }, cancellationToken);

        RequiredNullableDocument? result = await collection.GetAsync("a", cancellationToken: cancellationToken);
        Assert.NotNull(result);
        Assert.True(result.Embedding is null || result.Embedding.Value.IsEmpty);
    }

    [Fact]
    public async Task RequiredCustomVectorsCanUseAnEmptyValueFactory()
    {
        CancellationToken cancellationToken = TestContext.Current.CancellationToken;
        var definition = new VectorStoreCollectionDefinition
        {
            Properties =
            [
                new VectorStoreKeyProperty("Id", typeof(string)),
                new VectorStoreVectorProperty("Embedding", typeof(EncodedVector), 8)
            ]
        };
        TurboQuantRecordMapping<string, RequiredEncodedDocument> CreateMapping(Func<object?>? createEmptyVector) => new()
        {
            Definition = definition,
            RecordTypeInfo = TestJsonContext.Default.RequiredEncodedDocument,
            KeyTypeInfo = TestJsonContext.Default.String,
            GetKey = static document => document.Id,
            GetVector = static document => document.Embedding.ToArray(),
            CreateEmptyVector = createEmptyVector
        };

        var store = new InMemoryVectorStore();
        TurboQuantCollection<string, RequiredEncodedDocument> missingFactory =
            store.GetCollection("without-factory", CreateMapping(null));
        await missingFactory.UpsertAsync(
            new RequiredEncodedDocument { Id = "a", Embedding = new EncodedVector(Axis0) }, cancellationToken);
        NotSupportedException error = await Assert.ThrowsAsync<NotSupportedException>(
            () => missingFactory.GetAsync("a", cancellationToken: cancellationToken));
        Assert.Contains("CreateEmptyVector", error.Message);

        TurboQuantCollection<string, RequiredEncodedDocument> collection =
            store.GetCollection("with-factory", CreateMapping(static () => new EncodedVector([])));
        await collection.UpsertAsync(
            new RequiredEncodedDocument { Id = "a", Embedding = new EncodedVector(Axis0) }, cancellationToken);
        RequiredEncodedDocument? result = await collection.GetAsync("a", cancellationToken: cancellationToken);
        Assert.NotNull(result);
        Assert.Empty(result.Embedding);

        TurboQuantCollection<string, RequiredEncodedDocument> invalidFactory =
            store.GetCollection("invalid-factory", CreateMapping(static () => new EncodedVector([1f])));
        await invalidFactory.UpsertAsync(
            new RequiredEncodedDocument { Id = "a", Embedding = new EncodedVector(Axis0) }, cancellationToken);
        await Assert.ThrowsAsync<InvalidOperationException>(
            () => invalidFactory.GetAsync("a", cancellationToken: cancellationToken));
    }

    [Fact]
    public async Task RequiredConstructorVectorParametersCanBeExcluded()
    {
        CancellationToken cancellationToken = TestContext.Current.CancellationToken;
        var context = new TestJsonContext(new JsonSerializerOptions { RespectRequiredConstructorParameters = true });
        var mapping = new TurboQuantRecordMapping<string, ConstructorDocument>
        {
            Definition = new VectorStoreCollectionDefinition
            {
                Properties =
                [
                    new VectorStoreKeyProperty("Id", typeof(string)),
                    new VectorStoreVectorProperty("Embedding", typeof(ReadOnlyMemory<float>), 8)
                ]
            },
            RecordTypeInfo = context.ConstructorDocument,
            KeyTypeInfo = context.String,
            GetKey = static document => document.Id,
            GetVector = static document => document.Embedding
        };
        var store = new InMemoryVectorStore();
        TurboQuantCollection<string, ConstructorDocument> collection = store.GetCollection("constructor", mapping);
        await collection.UpsertAsync(new ConstructorDocument("a", Axis0), cancellationToken);

        ConstructorDocument? withoutVector = await collection.GetAsync("a", cancellationToken: cancellationToken);
        Assert.NotNull(withoutVector);
        Assert.True(withoutVector.Embedding.IsEmpty);
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

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task DynamicGuidKeysAreNotGeneratedWhenDisabled(bool omitKey)
    {
        var cancellationToken = TestContext.Current.CancellationToken;
        var definition = new VectorStoreCollectionDefinition
        {
            Properties =
            [
                new VectorStoreKeyProperty("Id", typeof(Guid)) { IsAutoGenerated = false },
                new VectorStoreVectorProperty("Embedding", typeof(float[]), 8)
            ]
        };
        var store = new InMemoryVectorStore();
        var collection = store.GetDynamicCollection("guids", definition);
        var record = new Dictionary<string, object?> { ["Embedding"] = Axis0 };
        if (!omitKey)
        {
            record["Id"] = null;
        }

        await Assert.ThrowsAsync<ArgumentException>(() => collection.UpsertAsync(record, cancellationToken));
        Assert.False(record.TryGetValue("Id", out var value) && value is not null);
        Assert.False(await collection.CollectionExistsAsync(cancellationToken));

        record["Id"] = Guid.Empty;
        await collection.UpsertAsync(record, cancellationToken);
        Assert.Equal(Guid.Empty, record["Id"]);
        Assert.NotNull(await collection.GetAsync(Guid.Empty, cancellationToken: cancellationToken));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public async Task DynamicGuidKeysAreGeneratedWhenEnabled(int keyKind)
    {
        var cancellationToken = TestContext.Current.CancellationToken;
        var definition = new VectorStoreCollectionDefinition
        {
            Properties =
            [
                new VectorStoreKeyProperty("Id", typeof(Guid)) { IsAutoGenerated = true },
                new VectorStoreVectorProperty("Embedding", typeof(float[]), 8)
            ]
        };
        var store = new InMemoryVectorStore();
        var collection = store.GetDynamicCollection("guids", definition);
        var record = new Dictionary<string, object?> { ["Embedding"] = Axis0 };
        if (keyKind != 0)
        {
            record["Id"] = keyKind == 2 ? Guid.Empty : null;
        }

        await collection.UpsertAsync(record, cancellationToken);

        var id = Assert.IsType<Guid>(record["Id"]);
        Assert.NotEqual(Guid.Empty, id);
        Assert.NotNull(await collection.GetAsync(id, cancellationToken: cancellationToken));
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
