using Microsoft.Extensions.VectorData;
using Microsoft.Extensions.VectorData.ProviderServices;
using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace Hasseware.Extensions.VectorData.TurboQuant;

public sealed class TurboQuantVectorStore : VectorStore
{
    private readonly string _directoryPath;
    private readonly int _bitWidth;
    private readonly JsonSerializerOptions _serializerOptions;
    internal static readonly ConcurrentDictionary<string, SemaphoreSlim> CollectionLocks = new(
        OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal);

    public TurboQuantVectorStore(string directoryPath, int bitWidth = 4, JsonSerializerOptions? serializerOptions = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directoryPath);
        if (bitWidth is not (2 or 3 or 4))
        {
            throw new ArgumentOutOfRangeException(nameof(bitWidth), "TurboQuant bit width must be 2, 3, or 4.");
        }

        _directoryPath = Path.GetFullPath(directoryPath);
        _bitWidth = bitWidth;
        _serializerOptions = serializerOptions is null
            ? new JsonSerializerOptions(JsonSerializerDefaults.General)
            : new JsonSerializerOptions(serializerOptions);
    }

    public override VectorStoreCollection<TKey, TRecord> GetCollection<TKey, TRecord>(
        string name,
        VectorStoreCollectionDefinition? definition = null)
        => new TurboQuantCollection<TKey, TRecord>(
            name, CollectionPath(name), _bitWidth, definition, _serializerOptions, CollectionLocks);

    public override VectorStoreCollection<object, Dictionary<string, object?>> GetDynamicCollection(
        string name,
        VectorStoreCollectionDefinition definition)
        => new TurboQuantCollection<object, Dictionary<string, object?>>(
            name, CollectionPath(name), _bitWidth, definition, _serializerOptions, CollectionLocks, dynamic: true);

    public override async IAsyncEnumerable<string> ListCollectionNamesAsync(
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (!Directory.Exists(_directoryPath))
        {
            yield break;
        }

        var names = new List<string>();
        foreach (string path in Directory.EnumerateFiles(_directoryPath, "*.tqv"))
        {
            cancellationToken.ThrowIfCancellationRequested();
            SemaphoreSlim gate = CollectionLocks.GetOrAdd(path, static _ => new SemaphoreSlim(1, 1));
            await gate.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                if (File.Exists(path))
                {
                    CollectionImage image = await CollectionStorage.ReadImageAsync(path, cancellationToken).ConfigureAwait(false);
                    names.Add(image.Name);
                }
            }
            finally
            {
                gate.Release();
            }
        }

        names.Sort(StringComparer.Ordinal);
        foreach (string name in names)
        {
            cancellationToken.ThrowIfCancellationRequested();
            yield return name;
        }
    }

    public override async Task<bool> CollectionExistsAsync(string name, CancellationToken cancellationToken = default)
    {
        string path = CollectionPath(name);
        if (!File.Exists(path))
        {
            return false;
        }

        SemaphoreSlim gate = CollectionLocks.GetOrAdd(path, static _ => new SemaphoreSlim(1, 1));
        await gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (!File.Exists(path))
            {
                return false;
            }

            CollectionImage image = await CollectionStorage.ReadImageAsync(path, cancellationToken).ConfigureAwait(false);
            return string.Equals(image.Name, name, StringComparison.Ordinal);
        }
        finally
        {
            gate.Release();
        }
    }

    public override async Task EnsureCollectionDeletedAsync(string name, CancellationToken cancellationToken = default)
    {
        string path = CollectionPath(name);
        SemaphoreSlim gate = CollectionLocks.GetOrAdd(path, static _ => new SemaphoreSlim(1, 1));
        await gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            File.Delete(path);
        }
        finally
        {
            gate.Release();
        }
    }

    public override object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        return serviceType.IsInstanceOfType(this) ? this : null;
    }

    private string CollectionPath(string name)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        byte[] nameBytes = Encoding.UTF8.GetBytes(name);
        return Path.Combine(_directoryPath, Convert.ToHexString(SHA256.HashData(nameBytes)) + ".tqv");
    }
}

internal sealed class TurboQuantModelBuilder : CollectionJsonModelBuilder
{
    public TurboQuantModelBuilder()
        : base(new CollectionModelBuildingOptions
        {
            SupportsMultipleVectors = false,
            RequiresAtLeastOneVector = true,
            UsesExternalSerializer = true
        })
    {
    }

    protected override void Customize()
    {
    }

    protected override bool IsDataPropertyTypeValid(Type type, out string errorMessage)
    {
        errorMessage = type.IsPointer || type.IsByRef
            ? $"Data property type '{type}' cannot be serialized."
            : string.Empty;
        return errorMessage.Length == 0;
    }

    protected override bool IsVectorPropertyTypeValid(Type type, out string errorMessage)
    {
        bool supported = type == typeof(float[]) ||
            type == typeof(Memory<float>) ||
            type == typeof(ReadOnlyMemory<float>) ||
            typeof(IEnumerable<float>).IsAssignableFrom(type);
        errorMessage = supported ? string.Empty : $"Vector property type '{type}' must be a float array, memory, or float sequence.";
        return supported;
    }
}

internal sealed class CollectionImage
{
    public required string Name { get; init; }
    public required string Schema { get; init; }
    public required byte[] Index { get; init; }
    public required Dictionary<string, ulong> KeyToId { get; init; }
    public required Dictionary<string, string> Records { get; init; }
    public ulong NextId { get; init; }
}

internal static class CollectionStorage
{
    public static async Task<CollectionImage> ReadImageAsync(string path, CancellationToken cancellationToken)
    {
        await using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read | FileShare.Delete, 64 * 1024, FileOptions.Asynchronous | FileOptions.SequentialScan);
        return await JsonSerializer.DeserializeAsync<CollectionImage>(stream, cancellationToken: cancellationToken).ConfigureAwait(false)
            ?? throw new InvalidDataException($"Collection image '{path}' is empty or invalid.");
    }

    public static async Task WriteImageAsync(string path, CollectionImage image, CancellationToken cancellationToken)
    {
        string directory = Path.GetDirectoryName(path)!;
        Directory.CreateDirectory(directory);
        string temporaryPath = Path.Combine(directory, $".{Path.GetFileName(path)}.{Guid.NewGuid():N}.tmp");
        try
        {
            await using (var stream = new FileStream(temporaryPath, FileMode.CreateNew, FileAccess.Write, FileShare.None, 64 * 1024, FileOptions.Asynchronous | FileOptions.WriteThrough))
            {
                await JsonSerializer.SerializeAsync(stream, image, cancellationToken: cancellationToken).ConfigureAwait(false);
                await stream.FlushAsync(cancellationToken).ConfigureAwait(false);
                stream.Flush(flushToDisk: true);
            }

            cancellationToken.ThrowIfCancellationRequested();
            File.Move(temporaryPath, path, overwrite: true);
        }
        finally
        {
            File.Delete(temporaryPath);
        }
    }
}
