using Microsoft.Extensions.VectorData;
using System.Collections.Concurrent;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization.Metadata;

namespace Hasseware.Extensions.VectorData.TurboQuant;

/// <summary>
/// Base class for TurboQuant-backed vector stores. Derived classes supply the persistence of
/// collection images (for example file system, blob storage, or database rows); the base class
/// implements collection management, quantized indexing, record mapping, and search.
/// </summary>
/// <remarks>
/// Each collection is persisted as a single self-describing image keyed by its collection name.
/// Implementations must replace images atomically, so readers observe either the previous or the
/// new committed image, and must return <see langword="null"/> from
/// <see cref="OpenCollectionReadStreamAsync"/> for collections that do not exist.
/// </remarks>
public abstract class TurboQuantVectorStore : VectorStore
{
    private const string TypedCollectionAotMessage =
        "VectorStore.GetCollection<TKey, TRecord> is annotated as requiring dynamic code. Use GetCollection(name, TurboQuantRecordMapping<TKey, TRecord>) for trimmed and Native AOT applications.";

    private readonly ConcurrentDictionary<string, SemaphoreSlim> _locks = new(StringComparer.Ordinal);
    private readonly ConcurrentDictionary<(Type Key, Type Record), object> _mappings = new();

    /// <summary>Initializes the store.</summary>
    /// <param name="bitWidth">TurboQuant bit width used for new collections: 2, 3, or 4.</param>
    /// <param name="serializerOptions">
    /// Options for dynamic collections. Additional data types must be resolvable through
    /// <see cref="JsonSerializerOptions.TypeInfoResolver"/>; common scalar and vector types are built in.
    /// </param>
    protected TurboQuantVectorStore(int bitWidth = 4, JsonSerializerOptions? serializerOptions = null)
    {
        if (bitWidth is not (2 or 3 or 4))
        {
            throw new ArgumentOutOfRangeException(nameof(bitWidth), "TurboQuant bit width must be 2, 3, or 4.");
        }

        BitWidth = bitWidth;
        var options = serializerOptions is null
            ? new JsonSerializerOptions(JsonSerializerDefaults.General)
            : new JsonSerializerOptions(serializerOptions);
        options.TypeInfoResolver = options.TypeInfoResolver is null
            ? TurboQuantDynamicJsonContext.Default
            : JsonTypeInfoResolver.Combine(options.TypeInfoResolver, TurboQuantDynamicJsonContext.Default);
        options.MakeReadOnly();
        SerializerOptions = options;
    }

    /// <summary>Gets the TurboQuant bit width used for new collections.</summary>
    public int BitWidth { get; }

    internal JsonSerializerOptions SerializerOptions { get; }

    /// <summary>
    /// Registers the explicit, trimming-safe mapping used by
    /// <see cref="GetCollection{TKey, TRecord}(string, VectorStoreCollectionDefinition?)"/> for a key/record type pair.
    /// </summary>
    /// <returns>This store, for chaining.</returns>
    public TurboQuantVectorStore RegisterRecordMapping<TKey, TRecord>(TurboQuantRecordMapping<TKey, TRecord> mapping)
        where TKey : notnull
        where TRecord : class
    {
        ArgumentNullException.ThrowIfNull(mapping);
        _ = new TypedRecordAccessor<TKey, TRecord>(mapping, definition: null);
        _mappings[(typeof(TKey), typeof(TRecord))] = mapping;
        return this;
    }

    /// <summary>Gets a typed collection using an explicit, trimming- and Native AOT-safe record mapping.</summary>
    public TurboQuantCollection<TKey, TRecord> GetCollection<TKey, TRecord>(string name, TurboQuantRecordMapping<TKey, TRecord> mapping)
        where TKey : notnull
        where TRecord : class
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(mapping);
        return new TurboQuantCollection<TKey, TRecord>(this, name, new TypedRecordAccessor<TKey, TRecord>(mapping, definition: null));
    }

    /// <summary>
    /// Gets a typed collection using the mapping registered with
    /// <see cref="RegisterRecordMapping{TKey, TRecord}"/>. A supplied <paramref name="definition"/>
    /// replaces the registered mapping's definition.
    /// </summary>
    [RequiresDynamicCode(TypedCollectionAotMessage)]
    [RequiresUnreferencedCode(TypedCollectionAotMessage)]
    public override VectorStoreCollection<TKey, TRecord> GetCollection<TKey, TRecord>(
        string name,
        VectorStoreCollectionDefinition? definition = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        if (!_mappings.TryGetValue((typeof(TKey), typeof(TRecord)), out object? registered))
        {
            throw new InvalidOperationException(
                $"No TurboQuant record mapping is registered for key type '{typeof(TKey)}' and record type '{typeof(TRecord)}'. " +
                "Call RegisterRecordMapping or use GetCollection(name, mapping).");
        }

        var mapping = (TurboQuantRecordMapping<TKey, TRecord>)registered;
        return new TurboQuantCollection<TKey, TRecord>(this, name, new TypedRecordAccessor<TKey, TRecord>(mapping, definition));
    }

    /// <inheritdoc />
    public override VectorStoreCollection<object, Dictionary<string, object?>> GetDynamicCollection(
        string name,
        VectorStoreCollectionDefinition definition)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(definition);
        return new TurboQuantCollection<object, Dictionary<string, object?>>(this, name, new DynamicRecordAccessor(definition, SerializerOptions));
    }

    /// <inheritdoc />
    public override async IAsyncEnumerable<string> ListCollectionNamesAsync(
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var names = new List<string>();
        await foreach (string name in ListStoredCollectionNamesAsync(cancellationToken).ConfigureAwait(false))
        {
            names.Add(name);
        }

        names.Sort(StringComparer.Ordinal);
        foreach (string name in names)
        {
            cancellationToken.ThrowIfCancellationRequested();
            yield return name;
        }
    }

    /// <inheritdoc />
    public override async Task<bool> CollectionExistsAsync(string name, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        SemaphoreSlim gate = GetCollectionLock(name);
        await gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            return await ImageExistsAsync(name, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            gate.Release();
        }
    }

    /// <inheritdoc />
    public override async Task EnsureCollectionDeletedAsync(string name, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        SemaphoreSlim gate = GetCollectionLock(name);
        await gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            await DeleteCollectionAsync(name, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            gate.Release();
        }
    }

    /// <inheritdoc />
    public override object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        return serviceType.IsInstanceOfType(this) ? this : null;
    }

    /// <summary>
    /// Opens a readable stream over the committed image of a collection, or returns
    /// <see langword="null"/> when no image exists. The caller disposes the stream.
    /// </summary>
    protected abstract ValueTask<Stream?> OpenCollectionReadStreamAsync(string name, CancellationToken cancellationToken);

    /// <summary>
    /// Writes and atomically commits a collection image. Implementations invoke
    /// <paramref name="writeImage"/> exactly once with a writable stream, and must commit the
    /// written bytes only when it completes successfully, preserving the previous image otherwise.
    /// </summary>
    protected abstract ValueTask WriteCollectionAsync(
        string name,
        Func<Stream, CancellationToken, ValueTask> writeImage,
        CancellationToken cancellationToken);

    /// <summary>Deletes the image of a collection. Deleting a missing collection is not an error.</summary>
    protected abstract ValueTask DeleteCollectionAsync(string name, CancellationToken cancellationToken);

    /// <summary>
    /// Enumerates the names of persisted collections in any order. Implementations without a name
    /// index can use <see cref="ReadCollectionNameAsync"/> to read the name from an image.
    /// </summary>
    protected abstract IAsyncEnumerable<string> ListStoredCollectionNamesAsync(CancellationToken cancellationToken);

    /// <summary>
    /// Gets the lock serializing operations on a collection. The default implementation coordinates
    /// operations within this store instance; override to share coordination across instances.
    /// It does not provide cross-process or distributed coordination.
    /// </summary>
    protected virtual SemaphoreSlim GetCollectionLock(string name)
        => _locks.GetOrAdd(name, static _ => new SemaphoreSlim(1, 1));

    /// <summary>Reads the collection name recorded in a collection image.</summary>
    protected static async ValueTask<string> ReadCollectionNameAsync(Stream image, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(image);
        CollectionImageHeader? header = await JsonSerializer.DeserializeAsync(
            image, CollectionImageJsonContext.Default.CollectionImageHeader, cancellationToken).ConfigureAwait(false);
        return header?.Name ?? throw new InvalidDataException("The collection image does not contain a collection name.");
    }

    internal SemaphoreSlim GetLock(string name) => GetCollectionLock(name);

    internal async ValueTask<bool> ImageExistsAsync(string name, CancellationToken cancellationToken)
    {
        Stream? stream = await OpenCollectionReadStreamAsync(name, cancellationToken).ConfigureAwait(false);
        if (stream is null)
        {
            return false;
        }

        await using (stream.ConfigureAwait(false))
        {
            string storedName = await ReadCollectionNameAsync(stream, cancellationToken).ConfigureAwait(false);
            return string.Equals(storedName, name, StringComparison.Ordinal);
        }
    }

    internal async ValueTask<CollectionImage?> ReadImageAsync(string name, CancellationToken cancellationToken)
    {
        Stream? stream = await OpenCollectionReadStreamAsync(name, cancellationToken).ConfigureAwait(false);
        if (stream is null)
        {
            return null;
        }

        await using (stream.ConfigureAwait(false))
        {
            return await JsonSerializer.DeserializeAsync(stream, CollectionImageJsonContext.Default.CollectionImage, cancellationToken).ConfigureAwait(false)
                ?? throw new InvalidDataException($"The image of collection '{name}' is empty or invalid.");
        }
    }

    internal ValueTask WriteImageAsync(CollectionImage image, CancellationToken cancellationToken)
        => WriteCollectionAsync(
            image.Name,
            (stream, token) => new ValueTask(JsonSerializer.SerializeAsync(stream, image, CollectionImageJsonContext.Default.CollectionImage, token)),
            cancellationToken);

    internal ValueTask DeleteImageAsync(string name, CancellationToken cancellationToken)
        => DeleteCollectionAsync(name, cancellationToken);
}
