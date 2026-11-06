using Microsoft.Extensions.VectorData;

namespace Hasseware.Extensions.VectorData.TurboQuant.Tests;

public sealed class TurboQuantVectorStoreTests
{
    [Fact]
    public async Task PersistsRecordsAndSupportsFilteredVectorSearch()
    {
        string directory = Path.Combine(Path.GetTempPath(), $"tq-vectordata-{Guid.NewGuid():N}");
        CancellationToken cancellationToken = TestContext.Current.CancellationToken;
        try
        {
            var store = new TurboQuantVectorStore(directory, bitWidth: 4);
            VectorStoreCollection<string, Document> collection = store.GetCollection<string, Document>("documents");
            await collection.EnsureCollectionExistsAsync(cancellationToken);
            await collection.UpsertAsync(
            [
                new Document("first", "manual", 1, new float[] { 1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f }),
                new Document("second", "guide", 2, new float[] { 0f, 1f, 0f, 0f, 0f, 0f, 0f, 0f }),
                new Document("third", "guide", 3, new float[] { 0f, 0f, 1f, 0f, 0f, 0f, 0f, 0f })
            ], cancellationToken);

            Document? first = await collection.GetAsync("first", new RecordRetrievalOptions { IncludeVectors = true }, cancellationToken);
            Assert.NotNull(first);
            Assert.Equal(1, first.Rank);
            Assert.Equal(8, first.Embedding.Length);
            Document? withoutVector = await collection.GetAsync("first", cancellationToken: cancellationToken);
            Assert.NotNull(withoutVector);
            Assert.Equal(0, withoutVector.Embedding.Length);

            var results = new List<VectorSearchResult<Document>>();
            await foreach (VectorSearchResult<Document> result in collection.SearchAsync(
                new ReadOnlyMemory<float>(new float[] { 0f, 1f, 0f, 0f, 0f, 0f, 0f, 0f }),
                top: 3,
                new VectorSearchOptions<Document>
                {
                    Filter = document => document.Category == "guide",
                    IncludeVectors = true
                },
                cancellationToken).WithCancellation(cancellationToken))
            {
                results.Add(result);
            }

            Assert.Equal(new[] { "second", "third" }, results.Select(result => result.Record.Id));
            Assert.All(results, result => Assert.Equal(8, result.Record.Embedding.Length));

            var reopenedStore = new TurboQuantVectorStore(directory, bitWidth: 4);
            VectorStoreCollection<string, Document> reopened = reopenedStore.GetCollection<string, Document>("documents");
            Assert.True(await reopenedStore.CollectionExistsAsync("documents", cancellationToken));
            var collectionNames = new List<string>();
            await foreach (string collectionName in reopenedStore.ListCollectionNamesAsync(cancellationToken).WithCancellation(cancellationToken))
            {
                collectionNames.Add(collectionName);
            }

            Assert.Equal(new[] { "documents" }, collectionNames);
            Document? persisted = await reopened.GetAsync("third", new RecordRetrievalOptions { IncludeVectors = true }, cancellationToken);
            Assert.NotNull(persisted);
            Assert.Equal(3, persisted.Rank);

            var filtered = new List<Document>();
            await foreach (Document record in reopened.GetAsync(
                document => document.Category == "guide",
                top: 1,
                new FilteredRecordRetrievalOptions<Document>
                {
                    IncludeVectors = true,
                    OrderBy = orderBy => orderBy.Descending(document => document.Rank)
                },
                cancellationToken).WithCancellation(cancellationToken))
            {
                filtered.Add(record);
            }

            Assert.Single(filtered);
            Assert.Equal("third", filtered[0].Id);
            Assert.True(await reopened.CollectionExistsAsync(cancellationToken));

            await reopened.DeleteAsync("second", cancellationToken);
            Assert.Null(await reopened.GetAsync("second", new RecordRetrievalOptions { IncludeVectors = true }, cancellationToken));
            await reopened.EnsureCollectionDeletedAsync(cancellationToken);
            Assert.False(await reopened.CollectionExistsAsync(cancellationToken));
        }
        finally
        {
            if (Directory.Exists(directory))
            {
                Directory.Delete(directory, recursive: true);
            }
        }
    }

    [Fact]
    public async Task CosineSimilarityIgnoresVectorMagnitude()
    {
        string directory = Path.Combine(Path.GetTempPath(), $"tq-vectordata-{Guid.NewGuid():N}");
        CancellationToken cancellationToken = TestContext.Current.CancellationToken;
        try
        {
            var store = new TurboQuantVectorStore(directory);
            VectorStoreCollection<string, CosineDocument> collection = store.GetCollection<string, CosineDocument>("cosine");
            await collection.UpsertAsync(
            [
                new CosineDocument { Id = "aligned", Embedding = new float[] { 1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f } },
                new CosineDocument { Id = "large-but-angled", Embedding = new float[] { 100f, 100f, 0f, 0f, 0f, 0f, 0f, 0f } }
            ], cancellationToken);

            await using IAsyncEnumerator<VectorSearchResult<CosineDocument>> results = collection.SearchAsync(
                new ReadOnlyMemory<float>(new float[] { 1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f }),
                top: 1,
                cancellationToken: cancellationToken).GetAsyncEnumerator(cancellationToken);

            Assert.True(await results.MoveNextAsync());
            Assert.Equal("aligned", results.Current.Record.Id);
        }
        finally
        {
            if (Directory.Exists(directory))
            {
                Directory.Delete(directory, recursive: true);
            }
        }
    }

    [Fact]
    public async Task SupportsDynamicRecordDefinitions()
    {
        string directory = Path.Combine(Path.GetTempPath(), $"tq-vectordata-{Guid.NewGuid():N}");
        CancellationToken cancellationToken = TestContext.Current.CancellationToken;
        try
        {
            var definition = new VectorStoreCollectionDefinition
            {
                Properties =
                [
                    new VectorStoreKeyProperty("Id", typeof(string)),
                    new VectorStoreDataProperty("Category", typeof(string)),
                    new VectorStoreVectorProperty("Embedding", typeof(ReadOnlyMemory<float>), 8)
                ]
            };
            var store = new TurboQuantVectorStore(directory);
            VectorStoreCollection<object, Dictionary<string, object?>> collection = store.GetDynamicCollection("dynamic", definition);
            await collection.UpsertAsync(new Dictionary<string, object?>
            {
                ["Id"] = "dynamic-1",
                ["Category"] = "notes",
                ["Embedding"] = new ReadOnlyMemory<float>(new float[] { 1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f })
            }, cancellationToken);

            Dictionary<string, object?>? record = await collection.GetAsync(
                "dynamic-1",
                new RecordRetrievalOptions { IncludeVectors = true },
                cancellationToken);

            Assert.NotNull(record);
            Assert.Equal("notes", record["Category"]);
            Assert.IsType<ReadOnlyMemory<float>>(record["Embedding"]);
        }
        finally
        {
            if (Directory.Exists(directory))
            {
                Directory.Delete(directory, recursive: true);
            }
        }
    }

    private sealed class Document(string id, string category, int rank, ReadOnlyMemory<float> embedding)
    {
        [VectorStoreKey]
        public string Id { get; set; } = id;

        [VectorStoreData]
        public string Category { get; set; } = category;

        [VectorStoreData]
        public int Rank { get; set; } = rank;

        [VectorStoreVector(8)]
        public ReadOnlyMemory<float> Embedding { get; set; } = embedding;
    }

    private sealed class CosineDocument
    {
        [VectorStoreKey]
        public string Id { get; set; } = "";

        [VectorStoreVector(8, DistanceFunction = DistanceFunction.CosineSimilarity)]
        public ReadOnlyMemory<float> Embedding { get; set; }
    }
}
