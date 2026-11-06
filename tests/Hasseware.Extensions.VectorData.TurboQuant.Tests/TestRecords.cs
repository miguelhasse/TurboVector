using Microsoft.Extensions.VectorData;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Text.Json.Serialization;

namespace Hasseware.Extensions.VectorData.TurboQuant.Tests;

internal sealed class Document(string id, string category, int rank, ReadOnlyMemory<float> embedding)
{
    [VectorStoreKey]
    public string Id { get; set; } = id;

    [VectorStoreData]
    [JsonPropertyName("category")]
    public string Category { get; set; } = category;

    [VectorStoreData]
    public int Rank { get; set; } = rank;

    [VectorStoreVector(8)]
    public ReadOnlyMemory<float> Embedding { get; set; } = embedding;
}

internal sealed class CosineDocument
{
    [VectorStoreKey]
    public string Id { get; set; } = "";

    [VectorStoreVector(8, DistanceFunction = DistanceFunction.CosineSimilarity)]
    public ReadOnlyMemory<float> Embedding { get; set; }
}

internal sealed class GuidDocument
{
    [VectorStoreKey]
    public Guid Id { get; set; }

    [VectorStoreVector(8)]
    public float[] Embedding { get; set; } = [];
}

[JsonSerializable(typeof(Document))]
[JsonSerializable(typeof(CosineDocument))]
[JsonSerializable(typeof(GuidDocument))]
[JsonSerializable(typeof(string))]
[JsonSerializable(typeof(Guid))]
internal sealed partial class TestJsonContext : JsonSerializerContext;

internal static class TestMappings
{
    public static TurboQuantRecordMapping<string, Document> Document { get; } = new()
    {
        Definition = new VectorStoreCollectionDefinition
        {
            Properties =
            [
                new VectorStoreKeyProperty(nameof(Tests.Document.Id), typeof(string)),
                new VectorStoreDataProperty(nameof(Tests.Document.Category), typeof(string)) { StorageName = "category" },
                new VectorStoreDataProperty(nameof(Tests.Document.Rank), typeof(int)),
                new VectorStoreVectorProperty(nameof(Tests.Document.Embedding), typeof(ReadOnlyMemory<float>), 8)
            ]
        },
        RecordTypeInfo = TestJsonContext.Default.Document,
        KeyTypeInfo = TestJsonContext.Default.String,
        GetKey = static document => document.Id,
        GetVector = static document => document.Embedding
    };

    public static TurboQuantRecordMapping<string, CosineDocument> CosineDocument { get; } = new()
    {
        Definition = new VectorStoreCollectionDefinition
        {
            Properties =
            [
                new VectorStoreKeyProperty(nameof(Tests.CosineDocument.Id), typeof(string)),
                new VectorStoreVectorProperty(nameof(Tests.CosineDocument.Embedding), typeof(ReadOnlyMemory<float>), 8)
                {
                    DistanceFunction = DistanceFunction.CosineSimilarity
                }
            ]
        },
        RecordTypeInfo = TestJsonContext.Default.CosineDocument,
        KeyTypeInfo = TestJsonContext.Default.String,
        GetKey = static document => document.Id,
        GetVector = static document => document.Embedding
    };

    public static TurboQuantRecordMapping<Guid, GuidDocument> GuidDocument { get; } = new()
    {
        Definition = new VectorStoreCollectionDefinition
        {
            Properties =
            [
                new VectorStoreKeyProperty(nameof(Tests.GuidDocument.Id), typeof(Guid)),
                new VectorStoreVectorProperty(nameof(Tests.GuidDocument.Embedding), typeof(float[]), 8)
            ]
        },
        RecordTypeInfo = TestJsonContext.Default.GuidDocument,
        KeyTypeInfo = TestJsonContext.Default.Guid,
        GetKey = static document => document.Id,
        SetKey = static (document, key) => document.Id = key,
        GetVector = static document => document.Embedding
    };
}

/// <summary>Non-file persistence used to verify the extensibility contract.</summary>
internal sealed class InMemoryVectorStore(int bitWidth = 4) : TurboQuantVectorStore(bitWidth)
{
    public ConcurrentDictionary<string, byte[]> Images { get; } = new(StringComparer.Ordinal);

    public bool FailWrites { get; set; }

    protected override ValueTask<Stream?> OpenCollectionReadStreamAsync(string name, CancellationToken cancellationToken)
        => ValueTask.FromResult<Stream?>(Images.TryGetValue(name, out byte[]? image) ? new MemoryStream(image, writable: false) : null);

    protected override async ValueTask WriteCollectionAsync(
        string name,
        Func<Stream, CancellationToken, ValueTask> writeImage,
        CancellationToken cancellationToken)
    {
        using var buffer = new MemoryStream();
        await writeImage(buffer, cancellationToken);
        if (FailWrites)
        {
            throw new IOException("Simulated write failure.");
        }

        Images[name] = buffer.ToArray();
    }

    protected override ValueTask DeleteCollectionAsync(string name, CancellationToken cancellationToken)
    {
        Images.TryRemove(name, out _);
        return ValueTask.CompletedTask;
    }

    protected override async IAsyncEnumerable<string> ListStoredCollectionNamesAsync(
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        foreach (string name in Images.Keys)
        {
            cancellationToken.ThrowIfCancellationRequested();
            await Task.Yield();
            yield return name;
        }
    }
}
