using Microsoft.Extensions.VectorData;
using Microsoft.Extensions.VectorData.ProviderServices;
using System.Numerics.Tensors;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using System.Text.Json.Serialization.Metadata;

namespace Hasseware.Extensions.VectorData.TurboQuant;

internal sealed class CollectionImage
{
    public required string Name { get; init; }
    public required string Schema { get; init; }
    public required byte[] Index { get; init; }
    public required Dictionary<string, ulong> KeyToId { get; init; }
    public required Dictionary<string, string> Records { get; init; }
    public ulong NextId { get; init; }
}

internal sealed class CollectionImageHeader
{
    public string? Name { get; init; }
}

[JsonSerializable(typeof(CollectionImage))]
[JsonSerializable(typeof(CollectionImageHeader))]
internal sealed partial class CollectionImageJsonContext : JsonSerializerContext;

[JsonSerializable(typeof(Dictionary<string, object?>))]
[JsonSerializable(typeof(object))]
[JsonSerializable(typeof(JsonElement))]
[JsonSerializable(typeof(string))]
[JsonSerializable(typeof(bool))]
[JsonSerializable(typeof(bool?))]
[JsonSerializable(typeof(byte))]
[JsonSerializable(typeof(byte?))]
[JsonSerializable(typeof(sbyte))]
[JsonSerializable(typeof(sbyte?))]
[JsonSerializable(typeof(short))]
[JsonSerializable(typeof(short?))]
[JsonSerializable(typeof(ushort))]
[JsonSerializable(typeof(ushort?))]
[JsonSerializable(typeof(int))]
[JsonSerializable(typeof(int?))]
[JsonSerializable(typeof(uint))]
[JsonSerializable(typeof(uint?))]
[JsonSerializable(typeof(long))]
[JsonSerializable(typeof(long?))]
[JsonSerializable(typeof(ulong))]
[JsonSerializable(typeof(ulong?))]
[JsonSerializable(typeof(float))]
[JsonSerializable(typeof(float?))]
[JsonSerializable(typeof(double))]
[JsonSerializable(typeof(double?))]
[JsonSerializable(typeof(decimal))]
[JsonSerializable(typeof(decimal?))]
[JsonSerializable(typeof(char))]
[JsonSerializable(typeof(char?))]
[JsonSerializable(typeof(Guid))]
[JsonSerializable(typeof(Guid?))]
[JsonSerializable(typeof(DateTime))]
[JsonSerializable(typeof(DateTime?))]
[JsonSerializable(typeof(DateTimeOffset))]
[JsonSerializable(typeof(DateTimeOffset?))]
[JsonSerializable(typeof(DateOnly))]
[JsonSerializable(typeof(DateOnly?))]
[JsonSerializable(typeof(TimeOnly))]
[JsonSerializable(typeof(TimeOnly?))]
[JsonSerializable(typeof(TimeSpan))]
[JsonSerializable(typeof(TimeSpan?))]
[JsonSerializable(typeof(float[]))]
[JsonSerializable(typeof(ReadOnlyMemory<float>))]
[JsonSerializable(typeof(ReadOnlyMemory<float>?))]
[JsonSerializable(typeof(Memory<float>))]
[JsonSerializable(typeof(Memory<float>?))]
[JsonSerializable(typeof(string[]))]
[JsonSerializable(typeof(List<string>))]
[JsonSerializable(typeof(int[]))]
[JsonSerializable(typeof(List<int>))]
[JsonSerializable(typeof(long[]))]
[JsonSerializable(typeof(List<long>))]
[JsonSerializable(typeof(double[]))]
[JsonSerializable(typeof(List<double>))]
[JsonSerializable(typeof(List<float>))]
internal sealed partial class TurboQuantDynamicJsonContext : JsonSerializerContext;

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
            type == typeof(Memory<float>?) ||
            type == typeof(ReadOnlyMemory<float>?) ||
            typeof(IEnumerable<float>).IsAssignableFrom(type);
        errorMessage = supported ? string.Empty : $"Vector property type '{type}' must be a float array, memory, or float sequence.";
        return supported;
    }
}

internal sealed class CollectionSchema
{
    public CollectionSchema(CollectionModel model, Func<PropertyModel, string> storageName)
    {
        Model = model;
        VectorProperty = model.VectorProperty;
        if (VectorProperty.EmbeddingGenerator is not null)
        {
            throw new NotSupportedException("TurboQuant stores precomputed vectors and does not support embedding generators.");
        }

        if (VectorProperty.IndexKind is not null and not IndexKind.QuantizedFlat)
        {
            throw new NotSupportedException($"TurboQuant does not support the index kind '{VectorProperty.IndexKind}'.");
        }

        if (VectorProperty.Dimensions is <= 0 or > 16384 || VectorProperty.Dimensions % 8 != 0)
        {
            throw new NotSupportedException(
                $"TurboQuant requires vector dimensions that are positive, divisible by 8 and no greater than 16384; '{VectorProperty.ModelName}' has {VectorProperty.Dimensions}.");
        }

        DistanceFunction = VectorProperty.DistanceFunction ?? Microsoft.Extensions.VectorData.DistanceFunction.DotProductSimilarity;
        if (DistanceFunction is not (Microsoft.Extensions.VectorData.DistanceFunction.DotProductSimilarity or
            Microsoft.Extensions.VectorData.DistanceFunction.NegativeDotProductSimilarity or
            Microsoft.Extensions.VectorData.DistanceFunction.CosineSimilarity or
            Microsoft.Extensions.VectorData.DistanceFunction.CosineDistance))
        {
            throw new NotSupportedException($"TurboQuant does not support the distance function '{DistanceFunction}'.");
        }

        NormalizesVectors = DistanceFunction is Microsoft.Extensions.VectorData.DistanceFunction.CosineSimilarity or
            Microsoft.Extensions.VectorData.DistanceFunction.CosineDistance;
        Fingerprint = CreateFingerprint(model, storageName);
    }

    public CollectionModel Model { get; }

    public VectorPropertyModel VectorProperty { get; }

    public string DistanceFunction { get; }

    public bool NormalizesVectors { get; }

    public string Fingerprint { get; }

    // The format must stay stable: persisted collections are matched against it on load.
    private static string CreateFingerprint(CollectionModel model, Func<PropertyModel, string> storageName)
    {
        var entries = new List<string>(model.Properties.Count);
        foreach (PropertyModel property in model.Properties)
        {
            string kind = property switch
            {
                KeyPropertyModel => "key",
                VectorPropertyModel vector => $"vector:{vector.Dimensions}:{vector.DistanceFunction}",
                _ => "data"
            };
            entries.Add($"{kind}|{property.ModelName}|{storageName(property)}|{property.Type.AssemblyQualifiedName}");
        }

        entries.Sort(StringComparer.Ordinal);
        return string.Join('\n', entries);
    }
}

internal abstract class RecordAccessor<TKey, TRecord>(CollectionSchema schema)
    where TKey : notnull
    where TRecord : class
{
    public CollectionSchema Schema { get; } = schema;

    public abstract string SerializeKey(TKey key);

    /// <summary>Returns the encoded key, assigning a generated key first when required.</summary>
    public abstract string PrepareKey(TRecord record);

    public abstract ReadOnlyMemory<float> GetVector(TRecord record);

    public abstract string Serialize(TRecord record);

    public abstract TRecord Deserialize(string json, bool includeVectors);

    protected static Exception KeyGenerationNotSupported(Type keyType)
        => new InvalidOperationException($"Automatic key generation is not supported for key type '{keyType}'.");
}

internal sealed class TypedRecordAccessor<TKey, TRecord> : RecordAccessor<TKey, TRecord>
    where TKey : notnull
    where TRecord : class
{
    private readonly TurboQuantRecordMapping<TKey, TRecord> _mapping;
    private readonly JsonPropertyInfo _vectorJsonProperty;
    private readonly Lazy<JsonTypeInfo>? _emptyVectorTypeInfo;

    public TypedRecordAccessor(TurboQuantRecordMapping<TKey, TRecord> mapping, VectorStoreCollectionDefinition? definition)
        : this(mapping, Create(mapping, definition ?? mapping.Definition))
    {
    }

    private TypedRecordAccessor(TurboQuantRecordMapping<TKey, TRecord> mapping, (CollectionSchema Schema, JsonPropertyInfo VectorJsonProperty) created)
        : base(created.Schema)
    {
        _mapping = mapping;
        _vectorJsonProperty = created.VectorJsonProperty;
        if (_vectorJsonProperty.IsRequired || mapping.RecordTypeInfo.Options.RespectRequiredConstructorParameters)
        {
            _emptyVectorTypeInfo = new Lazy<JsonTypeInfo>(CreateEmptyVectorTypeInfo);
        }
    }

    public override string SerializeKey(TKey key) => JsonSerializer.Serialize(key, _mapping.KeyTypeInfo);

    public override string PrepareKey(TRecord record)
    {
        TKey? key = _mapping.GetKey(record);
        if (key is null || Schema.Model.KeyProperty.IsAutoGenerated && key is Guid guid && guid == Guid.Empty)
        {
            if (typeof(TKey) != typeof(Guid))
            {
                throw KeyGenerationNotSupported(typeof(TKey));
            }

            Action<TRecord, TKey> setKey = _mapping.SetKey
                ?? throw new InvalidOperationException("The record mapping must define SetKey to assign automatically generated keys.");
            key = (TKey)(object)Guid.NewGuid();
            setKey(record, key);
        }

        return SerializeKey(key);
    }

    public override ReadOnlyMemory<float> GetVector(TRecord record) => _mapping.GetVector(record);

    public override string Serialize(TRecord record) => JsonSerializer.Serialize(record, _mapping.RecordTypeInfo);

    public override TRecord Deserialize(string json, bool includeVectors)
    {
        TRecord? record;
        if (includeVectors)
        {
            record = JsonSerializer.Deserialize(json, _mapping.RecordTypeInfo);
        }
        else
        {
            JsonNode? node = JsonNode.Parse(json);
            if (node is JsonObject obj)
            {
                if (_vectorJsonProperty.IsRequired || _mapping.RecordTypeInfo.Options.RespectRequiredConstructorParameters)
                {
                    obj[_vectorJsonProperty.Name] = EmptyVectorJson();
                }
                else
                {
                    obj.Remove(_vectorJsonProperty.Name);
                }
            }

            record = node.Deserialize(_mapping.RecordTypeInfo);
        }

        return record ?? throw new InvalidDataException("A persisted record deserialized to null.");
    }

    private static object? EmptyVectorValue(Type type)
    {
        if (type == typeof(ReadOnlyMemory<float>))
        {
            return ReadOnlyMemory<float>.Empty;
        }

        if (type == typeof(Memory<float>))
        {
            return Memory<float>.Empty;
        }

        return type.IsAssignableFrom(typeof(float[])) ? Array.Empty<float>() : null;
    }

    private JsonNode? EmptyVectorJson()
    {
        JsonTypeInfo typeInfo = _emptyVectorTypeInfo!.Value;
        Type type = _vectorJsonProperty.PropertyType;
        Func<object?>? createEmptyVector = _mapping.CreateEmptyVector;
        object? empty = createEmptyVector is null ? EmptyVectorValue(type) : createEmptyVector();
        if (empty is null && createEmptyVector is null && Nullable.GetUnderlyingType(type) is null)
        {
            try
            {
                empty = typeInfo.CreateObject?.Invoke() ?? JsonSerializer.Deserialize("[]", typeInfo);
            }
            catch (JsonException error)
            {
                throw new NotSupportedException(
                    $"The required vector property '{_vectorJsonProperty.Name}' cannot be created empty from its JSON metadata. " +
                    $"Set {nameof(TurboQuantRecordMapping<TKey, TRecord>.CreateEmptyVector)} on the record mapping.", error);
            }

            if (empty is null)
            {
                throw new NotSupportedException(
                    $"The required vector property '{_vectorJsonProperty.Name}' cannot be created empty from its JSON metadata. " +
                    $"Set {nameof(TurboQuantRecordMapping<TKey, TRecord>.CreateEmptyVector)} on the record mapping.");
            }
        }

        if (empty is not null && (!type.IsInstanceOfType(empty) || !IsEmptyVector(empty)))
        {
            throw new InvalidOperationException(
                $"The empty value for vector property '{_vectorJsonProperty.Name}' must be an empty '{type}'.");
        }

        return JsonSerializer.SerializeToNode(empty, typeInfo);
    }

    private static bool IsEmptyVector(object value)
    {
        switch (value)
        {
            case ReadOnlyMemory<float> readOnlyMemory:
                return readOnlyMemory.IsEmpty;
            case Memory<float> memory:
                return memory.IsEmpty;
            case ICollection<float> collection:
                return collection.Count == 0;
            case IReadOnlyCollection<float> collection:
                return collection.Count == 0;
            case IEnumerable<float> sequence:
                using (IEnumerator<float> enumerator = sequence.GetEnumerator())
                {
                    return !enumerator.MoveNext();
                }
            default:
                return false;
        }
    }

    private JsonTypeInfo CreateEmptyVectorTypeInfo()
    {
        JsonSerializerOptions options = _mapping.RecordTypeInfo.Options;
        if (_vectorJsonProperty.CustomConverter is { } converter)
        {
            options = new JsonSerializerOptions(options);
            options.Converters.Insert(0, converter);
            options.MakeReadOnly();
        }

        return options.GetTypeInfo(_vectorJsonProperty.PropertyType);
    }

    private static (CollectionSchema Schema, JsonPropertyInfo VectorJsonProperty) Create(
        TurboQuantRecordMapping<TKey, TRecord> mapping,
        VectorStoreCollectionDefinition definition)
    {
        ArgumentNullException.ThrowIfNull(mapping.Definition, nameof(mapping.Definition));
        ArgumentNullException.ThrowIfNull(mapping.RecordTypeInfo, nameof(mapping.RecordTypeInfo));
        ArgumentNullException.ThrowIfNull(mapping.KeyTypeInfo, nameof(mapping.KeyTypeInfo));
        ArgumentNullException.ThrowIfNull(mapping.GetKey, nameof(mapping.GetKey));
        ArgumentNullException.ThrowIfNull(mapping.GetVector, nameof(mapping.GetVector));

        JsonTypeInfo<TRecord> recordInfo = mapping.RecordTypeInfo;
        if (recordInfo.Kind != JsonTypeInfoKind.Object || recordInfo.Properties.Count == 0)
        {
            throw new InvalidOperationException(
                $"The JSON metadata for '{typeof(TRecord)}' must describe an object with properties. Use a source-generated JsonSerializerContext in metadata mode.");
        }

        CollectionModel model = new TurboQuantModelBuilder().BuildDynamic(definition, defaultEmbeddingGenerator: null, recordInfo.Options);
        if (model.KeyProperty.Type != typeof(TKey))
        {
            throw new InvalidOperationException(
                $"The key property '{model.KeyProperty.ModelName}' is declared as '{model.KeyProperty.Type}', but the mapping key type is '{typeof(TKey)}'.");
        }

        var storageNames = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (VectorStoreProperty property in definition.Properties)
        {
            string storageName = property.StorageName
                ?? recordInfo.Options.PropertyNamingPolicy?.ConvertName(property.Name)
                ?? property.Name;
            if (FindJsonProperty(recordInfo, storageName) is null)
            {
                throw new InvalidOperationException(
                    $"Property '{property.Name}' maps to JSON property '{storageName}', which is not present in the metadata for '{typeof(TRecord)}'. " +
                    "Set VectorStoreProperty.StorageName to the property's JSON name.");
            }

            storageNames[property.Name] = storageName;
        }

        var schema = new CollectionSchema(model, property => storageNames[property.ModelName]);
        JsonPropertyInfo vectorJsonProperty = FindJsonProperty(recordInfo, storageNames[model.VectorProperty.ModelName])!;
        return (schema, vectorJsonProperty);
    }

    private static JsonPropertyInfo? FindJsonProperty(JsonTypeInfo typeInfo, string name)
    {
        foreach (JsonPropertyInfo property in typeInfo.Properties)
        {
            if (string.Equals(property.Name, name, StringComparison.Ordinal))
            {
                return property;
            }
        }

        return null;
    }
}

internal sealed class DynamicRecordAccessor : RecordAccessor<object, Dictionary<string, object?>>
{
    private readonly JsonSerializerOptions _options;
    private readonly JsonTypeInfo<Dictionary<string, object?>> _recordInfo;
    private readonly JsonTypeInfo[] _propertyInfos;
    private readonly Dictionary<string, int>? _propertyIndicesByJsonName;
    private readonly HashSet<string>? _modelNames;

    public DynamicRecordAccessor(VectorStoreCollectionDefinition definition, JsonSerializerOptions options)
        : base(Create(definition, options))
    {
        _options = options;
        _recordInfo = (JsonTypeInfo<Dictionary<string, object?>>)options.GetTypeInfo(typeof(Dictionary<string, object?>));
        IReadOnlyList<PropertyModel> properties = Schema.Model.Properties;
        _propertyInfos = new JsonTypeInfo[properties.Count];
        if (options.DictionaryKeyPolicy is not null)
        {
            _propertyIndicesByJsonName = new Dictionary<string, int>(properties.Count, StringComparer.Ordinal);
            _modelNames = new HashSet<string>(properties.Count, StringComparer.Ordinal);
        }

        for (int i = 0; i < properties.Count; i++)
        {
            if (!options.TryGetTypeInfo(properties[i].Type, out JsonTypeInfo? typeInfo))
            {
                throw new NotSupportedException(
                    $"No JSON metadata is available for type '{properties[i].Type}' of property '{properties[i].ModelName}'. " +
                    "Add a JsonSerializerContext that includes it to the store's JsonSerializerOptions.TypeInfoResolver.");
            }

            _propertyInfos[i] = typeInfo;
            if (options.DictionaryKeyPolicy is { } policy)
            {
                string jsonName = policy.ConvertName(properties[i].ModelName)
                    ?? throw new InvalidOperationException("The dictionary key policy returned a null property name.");
                if (!_propertyIndicesByJsonName!.TryAdd(jsonName, i))
                {
                    throw new InvalidOperationException($"The dictionary key policy maps multiple properties to '{jsonName}'.");
                }

                _modelNames!.Add(properties[i].ModelName);
            }
        }
    }

    public override string SerializeKey(object key) => JsonSerializer.Serialize(key, _options.GetTypeInfo(key.GetType()));

    public override string PrepareKey(Dictionary<string, object?> record)
    {
        KeyPropertyModel keyProperty = Schema.Model.KeyProperty;
        object? key = keyProperty.GetValueAsObject(record);
        if (key is null || keyProperty.IsAutoGenerated && key is Guid guid && guid == Guid.Empty)
        {
            if (keyProperty.Type != typeof(Guid))
            {
                throw KeyGenerationNotSupported(keyProperty.Type);
            }

            key = Guid.NewGuid();
            keyProperty.SetValueAsObject(record, key);
        }

        return SerializeKey(key);
    }

    public override ReadOnlyMemory<float> GetVector(Dictionary<string, object?> record)
        => VectorValues.ToMemory(Schema.VectorProperty.GetValueAsObject(record), Schema.VectorProperty.ModelName);

    public override string Serialize(Dictionary<string, object?> record)
    {
        if (_options.DictionaryKeyPolicy is { } policy)
        {
            var jsonNames = new HashSet<string>(record.Count, StringComparer.Ordinal);
            IReadOnlyList<PropertyModel> properties = Schema.Model.Properties;
            foreach (string modelName in record.Keys)
            {
                string jsonName = policy.ConvertName(modelName)
                    ?? throw new InvalidOperationException("The dictionary key policy returned a null property name.");
                bool conflictsWithModelName = _propertyIndicesByJsonName!.TryGetValue(jsonName, out int index)
                    ? !string.Equals(properties[index].ModelName, modelName, StringComparison.Ordinal)
                    : _modelNames!.Contains(jsonName);
                if (!jsonNames.Add(jsonName) || conflictsWithModelName)
                {
                    throw new ArgumentException($"The dictionary key policy creates an ambiguous property name '{jsonName}'.", nameof(record));
                }
            }
        }

        return JsonSerializer.Serialize(record, _recordInfo);
    }

    public override Dictionary<string, object?> Deserialize(string json, bool includeVectors)
    {
        Dictionary<string, object?> record = JsonSerializer.Deserialize(json, _recordInfo)
            ?? throw new InvalidDataException("A persisted record deserialized to null.");
        IReadOnlyList<PropertyModel> properties = Schema.Model.Properties;
        if (_propertyIndicesByJsonName is not null)
        {
            var normalized = new Dictionary<string, object?>(record.Count, StringComparer.Ordinal);
            foreach ((string jsonName, object? value) in record)
            {
                string modelName = jsonName;
                object? converted = value;
                if (_propertyIndicesByJsonName.TryGetValue(jsonName, out int index))
                {
                    PropertyModel property = properties[index];
                    if (!includeVectors && property is VectorPropertyModel)
                    {
                        continue;
                    }

                    modelName = property.ModelName;
                    if (value is JsonElement element)
                    {
                        converted = element.Deserialize(_propertyInfos[index]);
                    }
                }

                if (!normalized.TryAdd(modelName, converted))
                {
                    throw new InvalidDataException($"The persisted record contains an ambiguous property name '{modelName}'.");
                }
            }

            return normalized;
        }

        for (int i = 0; i < properties.Count; i++)
        {
            PropertyModel property = properties[i];
            if (!includeVectors && property is VectorPropertyModel)
            {
                record.Remove(property.ModelName);
                continue;
            }

            if (record.TryGetValue(property.ModelName, out object? value) && value is JsonElement element)
            {
                record[property.ModelName] = element.Deserialize(_propertyInfos[i]);
            }
        }

        return record;
    }

    private static CollectionSchema Create(VectorStoreCollectionDefinition definition, JsonSerializerOptions options)
    {
        ArgumentNullException.ThrowIfNull(definition);
        CollectionModel model = new TurboQuantModelBuilder().BuildDynamic(definition, defaultEmbeddingGenerator: null, options);
        return new CollectionSchema(model, static property => property.StorageName);
    }
}

internal static class VectorValues
{
    public static ReadOnlyMemory<float> ToMemory(object? value, string propertyName)
        => value switch
        {
            float[] array => array,
            ReadOnlyMemory<float> memory => memory,
            Memory<float> memory => memory,
            IEnumerable<float> enumerable => enumerable.ToArray(),
            _ => throw new ArgumentException($"Vector property '{propertyName}' must contain float coordinates.")
        };

    public static void Validate(ReadOnlySpan<float> vector, string propertyName, int? dimensions)
    {
        if (vector.IsEmpty)
        {
            throw new ArgumentException($"Vector property '{propertyName}' cannot be empty.");
        }

        // MaxMagnitude propagates NaN, so this rejects NaN, infinities, and |x| >= 1e16.
        if (!(MathF.Abs(TensorPrimitives.MaxMagnitude(vector)) < 1e16f))
        {
            throw new ArgumentException($"Vector property '{propertyName}' contains an invalid coordinate.");
        }

        if (dimensions is int expected && vector.Length != expected)
        {
            throw new ArgumentException($"Vector property '{propertyName}' requires {expected} dimensions, but received {vector.Length}.");
        }
    }

    public static void Normalize(Span<float> vector)
    {
        float norm = TensorPrimitives.Norm(vector);
        if (norm > 1e-10f)
        {
            TensorPrimitives.Divide(vector, norm, vector);
        }
        else
        {
            vector.Clear();
        }
    }
}
