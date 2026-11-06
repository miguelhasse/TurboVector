# Hasseware.Extensions.VectorData.TurboQuant

A file-backed [Microsoft.Extensions.VectorData](https://learn.microsoft.com/dotnet/ai/vector-databases) provider using TurboVector's compressed `IdMapIndex`.

## Usage

```csharp
using Hasseware.Extensions.VectorData.TurboQuant;
using Microsoft.Extensions.VectorData;

var store = new TurboQuantVectorStore("./vector-data", bitWidth: 4);
var articles = store.GetCollection<string, Article>("articles");

await articles.EnsureCollectionExistsAsync();
await articles.UpsertAsync(new Article
{
    Id = "article-1",
    Category = "documentation",
    Embedding = new ReadOnlyMemory<float>(embedding)
});

await foreach (var result in articles.SearchAsync(
    new ReadOnlyMemory<float>(queryEmbedding),
    top: 5,
    new VectorSearchOptions<Article>
    {
        Filter = article => article.Category == "documentation"
    }))
{
    Console.WriteLine($"{result.Record.Id}: {result.Score}");
}

public sealed class Article
{
    [VectorStoreKey]
    public string Id { get; set; } = "";

    [VectorStoreData]
    public string Category { get; set; } = "";

    [VectorStoreVector(128)]
    public ReadOnlyMemory<float> Embedding { get; set; }
}
```

The vector property must use a supported float vector type (`ReadOnlyMemory<float>`, `Memory<float>`, `float[]`, or an `IEnumerable<float>` implementation). Vectors must have the declared dimension when one is specified; TurboVector also requires dimensions to be positive multiples of 8. This provider stores precomputed vectors and does not generate embeddings from text.

## Persistence and behavior

- Each named collection is stored as one JSON image containing the TurboVector index, key-to-ID mapping, and serialized records. Updates write a sibling temporary file and atomically replace the collection image.
- Collections are loaded lazily when accessed. `EnsureCollectionExistsAsync` creates an empty collection; reopening the store with the same directory and schema restores its records.
- `TurboQuantVectorStore` defaults to 4-bit quantization. Supported widths are 2, 3, and 4 bits.
- Predicate filters and filtered-retrieval ordering are evaluated against loaded records in process.
- The default score is TurboVector's dot-product similarity. Negative dot-product similarity and cosine similarity/distance are also supported. Cosine modes normalize vectors during indexing and search; other distance functions are rejected.
- Search accepts float vectors directly. `IncludeVectors` controls whether returned records retain vector values.
- Concurrent operations within this process are serialized per collection. Do not write to the same collection directory from multiple processes concurrently.

The provider targets .NET 10 and references `Microsoft.Extensions.VectorData.Abstractions`.

## Tests

Provider tests live in a separate assembly. Run them from the solution root:

```bash
dotnet test tests/Hasseware.Extensions.VectorData.TurboQuant.Tests
```
