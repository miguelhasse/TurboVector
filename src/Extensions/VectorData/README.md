# Hasseware.Extensions.VectorData.TurboQuant

A [Microsoft.Extensions.VectorData](https://learn.microsoft.com/dotnet/ai/vector-databases) provider using TurboVector's compressed `IdMapIndex`. The provider and the TurboVector core library are trimming and Native AOT compatible.

## Usage

`TurboQuantFileSystemVectorStore` persists each collection as a file in a directory. Typed collections use an explicit `TurboQuantRecordMapping` backed by source-generated `System.Text.Json` metadata, so no reflection is required:

```csharp
using Hasseware.Extensions.VectorData.TurboQuant;
using Microsoft.Extensions.VectorData;
using System.Text.Json.Serialization;

var store = new TurboQuantFileSystemVectorStore("./vector-data", bitWidth: 4);
var articles = store.GetCollection("articles", ArticleMapping.Instance);

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
    public string Id { get; set; } = "";
    public string Category { get; set; } = "";
    public ReadOnlyMemory<float> Embedding { get; set; }
}

[JsonSerializable(typeof(Article))]
[JsonSerializable(typeof(string))]
internal sealed partial class ArticleJsonContext : JsonSerializerContext;

internal static class ArticleMapping
{
    public static TurboQuantRecordMapping<string, Article> Instance { get; } = new()
    {
        Definition = new VectorStoreCollectionDefinition
        {
            Properties =
            [
                new VectorStoreKeyProperty(nameof(Article.Id), typeof(string)),
                new VectorStoreDataProperty(nameof(Article.Category), typeof(string)),
                new VectorStoreVectorProperty(nameof(Article.Embedding), typeof(ReadOnlyMemory<float>), 128)
            ]
        },
        RecordTypeInfo = ArticleJsonContext.Default.Article,
        KeyTypeInfo = ArticleJsonContext.Default.String,
        GetKey = static article => article.Id,
        GetVector = static article => article.Embedding
    };
}
```

- `SetKey` is optional; supply it for `Guid` keys so `UpsertAsync` can assign generated keys.
- If a property's JSON name differs from its model name (for example via `[JsonPropertyName]` or a naming policy that is not applied through the definition), set `StorageName` on the definition property. Mappings are validated against the `JsonTypeInfo` when a collection is created.
- To use the standard `VectorStore.GetCollection<TKey, TRecord>(name)` API, register the mapping once: `store.RegisterRecordMapping(ArticleMapping.Instance)`. That API is annotated as requiring dynamic code by Microsoft.Extensions.VectorData itself, so AOT applications should prefer the `GetCollection(name, mapping)` overload. Unregistered record types throw `InvalidOperationException`.
- `GetDynamicCollection(name, definition)` is AOT-safe. Property types other than primitives, `Guid`, date/time types, `float[]` and `ReadOnlyMemory<float>` need JSON metadata supplied through the `serializerOptions` constructor argument (for example `new JsonSerializerOptions { TypeInfoResolver = MyJsonContext.Default }`).

The vector property must use a supported float vector type (`ReadOnlyMemory<float>`, `Memory<float>`, `float[]`, or an `IEnumerable<float>` implementation). TurboVector requires dimensions to be positive multiples of 8 (at most 16384); other dimensions are rejected when the collection is created. This provider stores precomputed vectors and does not generate embeddings from text.

## Custom persistence backends

`TurboQuantVectorStore` is an abstract base class that owns collection management, record mapping, quantization and search. Backends such as blob storage or a database only implement storage of opaque collection images:

```csharp
public sealed class BlobVectorStore(BlobContainerClient container) : TurboQuantVectorStore(bitWidth: 4)
{
    protected override async ValueTask<Stream?> OpenCollectionReadStreamAsync(string name, CancellationToken cancellationToken) { /* null when missing */ }
    protected override async ValueTask WriteCollectionAsync(string name, Func<Stream, CancellationToken, ValueTask> writeImage, CancellationToken cancellationToken) { /* write, then commit atomically */ }
    protected override async ValueTask DeleteCollectionAsync(string name, CancellationToken cancellationToken) { /* no-op when missing */ }
    protected override async IAsyncEnumerable<string> ListStoredCollectionNamesAsync([EnumeratorCancellation] CancellationToken cancellationToken) { /* collection names */ }
}
```

Contract:

- `OpenCollectionReadStreamAsync` returns `null` when the collection does not exist. The caller disposes the stream.
- `WriteCollectionAsync` must invoke `writeImage` exactly once and make the new image visible atomically: if writing or committing fails, the previously committed image must stay readable.
- `ListStoredCollectionNamesAsync` yields the original collection names (not storage keys). `ReadCollectionNameAsync(stream)` reads the name from a stored image when the backend only knows its keys.
- Operations on a collection are serialized through `GetCollectionLock(name)`; the default lock is per store instance. Override it to share locks between instances, or use backend concurrency control (for example ETags) for multi-process deployments.

## Persistence and behavior

- Each named collection is stored as one JSON image containing the TurboVector index, key-to-ID mapping, and serialized records. Images written by earlier versions of the provider remain readable as long as the record schema is unchanged.
- `TurboQuantFileSystemVectorStore` names files by the SHA-256 of the collection name, writes a sibling temporary file and atomically replaces the image. Store instances in the same process share per-file locks. Do not write to the same directory from multiple processes concurrently.
- Collections are loaded lazily when accessed. `EnsureCollectionExistsAsync` creates an empty collection; reopening the store with the same storage and schema restores its records.
- The store defaults to 4-bit quantization. Supported widths are 2, 3, and 4 bits.
- Predicate filters and filtered-retrieval ordering are evaluated against loaded records in process.
- The default score is TurboVector's dot-product similarity. Negative dot-product similarity and cosine similarity/distance are also supported. Cosine modes normalize copies of vectors during indexing and search; caller-supplied arrays are not modified. Other distance functions are rejected.
- Search accepts float vectors directly. `IncludeVectors` controls whether returned records retain vector values.

## Migrating from earlier versions

`TurboQuantVectorStore` is now abstract. Replace `new TurboQuantVectorStore(directory)` with `new TurboQuantFileSystemVectorStore(directory)` and either pass a `TurboQuantRecordMapping` to `GetCollection` or register it with `RegisterRecordMapping`. Existing collection files are reused unchanged.

The provider targets .NET 10 and references `Microsoft.Extensions.VectorData.Abstractions`.

## Tests

Provider tests live in a separate assembly. Run them from the solution root:

```bash
dotnet test tests/Hasseware.Extensions.VectorData.TurboQuant.Tests
```
