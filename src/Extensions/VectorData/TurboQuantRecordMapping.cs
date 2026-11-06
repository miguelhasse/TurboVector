using Microsoft.Extensions.VectorData;
using System.Text.Json.Serialization.Metadata;

namespace Hasseware.Extensions.VectorData.TurboQuant;

/// <summary>
/// Explicit, reflection-free mapping between a record type and a TurboQuant collection. Use with
/// source-generated JSON metadata to keep typed collections trimming- and Native AOT-compatible.
/// </summary>
/// <remarks>
/// <see cref="VectorStoreProperty.StorageName"/> must be the JSON name of each property in
/// <see cref="RecordTypeInfo"/> when it differs from the naming-policy-converted model name.
/// Collections created by earlier versions remain readable when the definition matches the
/// record's original attributes (names, types, dimensions, distance function, and JSON names).
/// </remarks>
public sealed class TurboQuantRecordMapping<TKey, TRecord>
    where TKey : notnull
    where TRecord : class
{
    /// <summary>Gets the key, data, and single vector property definitions.</summary>
    public required VectorStoreCollectionDefinition Definition { get; init; }

    /// <summary>Gets the generated JSON metadata used to persist records.</summary>
    public required JsonTypeInfo<TRecord> RecordTypeInfo { get; init; }

    /// <summary>Gets the generated JSON metadata used to encode keys.</summary>
    public required JsonTypeInfo<TKey> KeyTypeInfo { get; init; }

    /// <summary>Gets the accessor returning the record key.</summary>
    public required Func<TRecord, TKey?> GetKey { get; init; }

    /// <summary>Gets the accessor returning the record vector.</summary>
    public required Func<TRecord, ReadOnlyMemory<float>> GetVector { get; init; }

    /// <summary>Gets the optional key setter used to assign auto-generated <see cref="Guid"/> keys.</summary>
    public Action<TRecord, TKey>? SetKey { get; init; }
}
