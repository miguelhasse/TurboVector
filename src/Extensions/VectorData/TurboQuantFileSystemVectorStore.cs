using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace Hasseware.Extensions.VectorData.TurboQuant;

/// <summary>
/// TurboQuant vector store persisting each collection as a <c>.tqv</c> file in a directory.
/// </summary>
/// <remarks>
/// File names are the SHA-256 of the UTF-8 collection name. Writes go to a temporary file in the
/// same directory, are flushed to disk, and then replace the committed file. Operations on the
/// same file are serialized across store instances in the current process.
/// </remarks>
public sealed class TurboQuantFileSystemVectorStore : TurboQuantVectorStore
{
    private const int BufferSize = 64 * 1024;

    private static readonly ConcurrentDictionary<string, SemaphoreSlim> FileLocks = new(
        OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal);

    /// <summary>Initializes a store rooted at <paramref name="directoryPath"/>.</summary>
    public TurboQuantFileSystemVectorStore(string directoryPath, int bitWidth = 4, JsonSerializerOptions? serializerOptions = null)
        : base(bitWidth, serializerOptions)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directoryPath);
        DirectoryPath = Path.GetFullPath(directoryPath);
    }

    /// <summary>Gets the full path of the directory containing collection files.</summary>
    public string DirectoryPath { get; }

    /// <inheritdoc />
    protected override SemaphoreSlim GetCollectionLock(string name) => GetFileLock(CollectionPath(name));

    /// <inheritdoc />
    protected override ValueTask<Stream?> OpenCollectionReadStreamAsync(string name, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.FromResult<Stream?>(OpenRead(CollectionPath(name)));
    }

    /// <inheritdoc />
    protected override async ValueTask WriteCollectionAsync(
        string name,
        Func<Stream, CancellationToken, ValueTask> writeImage,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(writeImage);
        string path = CollectionPath(name);
        Directory.CreateDirectory(DirectoryPath);
        string temporaryPath = Path.Combine(DirectoryPath, $".{Path.GetFileName(path)}.{Guid.NewGuid():N}.tmp");
        try
        {
            var stream = new FileStream(temporaryPath, FileMode.CreateNew, FileAccess.Write, FileShare.None, BufferSize, FileOptions.Asynchronous | FileOptions.WriteThrough);
            await using (stream.ConfigureAwait(false))
            {
                await writeImage(stream, cancellationToken).ConfigureAwait(false);
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

    /// <inheritdoc />
    protected override ValueTask DeleteCollectionAsync(string name, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        try
        {
            File.Delete(CollectionPath(name));
        }
        catch (DirectoryNotFoundException)
        {
        }

        return ValueTask.CompletedTask;
    }

    /// <inheritdoc />
    protected override async IAsyncEnumerable<string> ListStoredCollectionNamesAsync(
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        if (!Directory.Exists(DirectoryPath))
        {
            yield break;
        }

        foreach (string path in Directory.EnumerateFiles(DirectoryPath, "*.tqv"))
        {
            cancellationToken.ThrowIfCancellationRequested();
            SemaphoreSlim gate = GetFileLock(path);
            await gate.WaitAsync(cancellationToken).ConfigureAwait(false);
            string? name = null;
            try
            {
                Stream? stream = OpenRead(path);
                if (stream is not null)
                {
                    await using (stream.ConfigureAwait(false))
                    {
                        name = await ReadCollectionNameAsync(stream, cancellationToken).ConfigureAwait(false);
                    }
                }
            }
            finally
            {
                gate.Release();
            }

            if (name is not null)
            {
                yield return name;
            }
        }
    }

    private static SemaphoreSlim GetFileLock(string path)
        => FileLocks.GetOrAdd(path, static _ => new SemaphoreSlim(1, 1));

    private static FileStream? OpenRead(string path)
    {
        try
        {
            return new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read | FileShare.Delete, BufferSize, FileOptions.Asynchronous | FileOptions.SequentialScan);
        }
        catch (Exception exception) when (exception is FileNotFoundException or DirectoryNotFoundException)
        {
            return null;
        }
    }

    private string CollectionPath(string name)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        return Path.Combine(DirectoryPath, Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(name))) + ".tqv");
    }
}
