using Microsoft.Extensions.VectorData;
using Microsoft.Extensions.VectorData.ProviderServices;
using System.Globalization;
using System.Linq.Expressions;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using TurboVector;

namespace Hasseware.Extensions.VectorData.TurboQuant;

public sealed class TurboQuantCollection<TKey, TRecord> : VectorStoreCollection<TKey, TRecord>
    where TKey : notnull
    where TRecord : class
{
    private readonly string _path;
    private readonly int _bitWidth;
    private readonly JsonSerializerOptions _serializerOptions;
    private readonly CollectionModel _model;
    private readonly string _schema;
    private readonly string _distanceFunction;
    private readonly SemaphoreSlim _gate;

    internal TurboQuantCollection(
        string name,
        string path,
        int bitWidth,
        VectorStoreCollectionDefinition? definition,
        JsonSerializerOptions serializerOptions,
        System.Collections.Concurrent.ConcurrentDictionary<string, SemaphoreSlim> locks,
        bool dynamic = false)
        : base()
    {
        _path = path;
        _bitWidth = bitWidth;
        _serializerOptions = serializerOptions;
        Name = name;
        _gate = locks.GetOrAdd(path, static _ => new SemaphoreSlim(1, 1));
        var builder = new TurboQuantModelBuilder();
        _model = dynamic
            ? builder.BuildDynamic(definition ?? throw new ArgumentNullException(nameof(definition)), defaultEmbeddingGenerator: null, serializerOptions)
            : builder.Build(typeof(TRecord), typeof(TKey), definition, defaultEmbeddingGenerator: null, serializerOptions);
        _distanceFunction = _model.VectorProperty.DistanceFunction ?? DistanceFunction.DotProductSimilarity;
        if (_model.VectorProperty.EmbeddingGenerator is not null)
        {
            throw new NotSupportedException("TurboQuant stores precomputed vectors and does not support embedding generators.");
        }

        if (_model.VectorProperty.IndexKind is not null and not IndexKind.QuantizedFlat)
        {
            throw new NotSupportedException($"TurboQuant does not support the index kind '{_model.VectorProperty.IndexKind}'.");
        }

        if (_distanceFunction is not (DistanceFunction.DotProductSimilarity or
            DistanceFunction.NegativeDotProductSimilarity or
            DistanceFunction.CosineSimilarity or
            DistanceFunction.CosineDistance))
        {
            throw new NotSupportedException($"TurboQuant does not support the distance function '{_distanceFunction}'.");
        }

        _schema = CreateSchemaFingerprint(_model);
    }

    public override string Name { get; }

    public override async Task<bool> CollectionExistsAsync(CancellationToken cancellationToken = default)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            return File.Exists(_path);
        }
        finally
        {
            _gate.Release();
        }
    }

    public override async Task EnsureCollectionExistsAsync(CancellationToken cancellationToken = default)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (!File.Exists(_path))
            {
                State state = NewState();
                await SaveAsync(state, cancellationToken).ConfigureAwait(false);
            }
            else
            {
                _ = await LoadAsync(cancellationToken).ConfigureAwait(false);
            }
        }
        finally
        {
            _gate.Release();
        }
    }

    public override async Task EnsureCollectionDeletedAsync(CancellationToken cancellationToken = default)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            File.Delete(_path);
        }
        finally
        {
            _gate.Release();
        }
    }

    public override async Task<TRecord?> GetAsync(
        TKey key,
        RecordRetrievalOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(key);
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            State? state = await LoadIfExistsAsync(cancellationToken).ConfigureAwait(false);
            if (state is null || !state.KeyToId.TryGetValue(SerializeKey(key), out ulong id) ||
                !state.Records.TryGetValue(IdKey(id), out string? json))
            {
                return default;
            }

            TRecord record = DeserializeRecord(json);
            return options?.IncludeVectors == true ? record : WithoutVectors(record);
        }
        finally
        {
            _gate.Release();
        }
    }

    public override async IAsyncEnumerable<TRecord> GetAsync(
        Expression<Func<TRecord, bool>> filter,
        int top,
        FilteredRecordRetrievalOptions<TRecord>? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(filter);
        ArgumentOutOfRangeException.ThrowIfNegative(top);
        if (top == 0)
        {
            yield break;
        }

        int requestedSkip = options?.Skip ?? 0;
        ArgumentOutOfRangeException.ThrowIfNegative(requestedSkip);
        List<TRecord> result = [];
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            State? state = await LoadIfExistsAsync(cancellationToken).ConfigureAwait(false);
            if (state is null)
            {
                yield break;
            }

            Func<TRecord, bool> predicate = filter.Compile();
            foreach (string json in state.Records.Values)
            {
                cancellationToken.ThrowIfCancellationRequested();
                TRecord record = DeserializeRecord(json);
                if (!predicate(record))
                {
                    continue;
                }

                result.Add(options?.IncludeVectors == true ? record : WithoutVectors(record));
            }
        }
        finally
        {
            _gate.Release();
        }

        IEnumerable<TRecord> ordered = ApplyOrdering(result, options?.OrderBy);
        foreach (TRecord record in ordered.Skip(requestedSkip).Take(top))
        {
            cancellationToken.ThrowIfCancellationRequested();
            yield return record;
        }
    }

    public override async Task DeleteAsync(TKey key, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(key);
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            State? state = await LoadIfExistsAsync(cancellationToken).ConfigureAwait(false);
            if (state is null)
            {
                return;
            }

            string keyJson = SerializeKey(key);
            if (!state.KeyToId.Remove(keyJson, out ulong id))
            {
                return;
            }

            state.Records.Remove(IdKey(id));
            state.Index.Remove(id);
            await SaveAsync(state, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _gate.Release();
        }
    }

    public override async Task UpsertAsync(TRecord record, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(record);
        await UpsertAsync([record], cancellationToken).ConfigureAwait(false);
    }

    public override async Task UpsertAsync(IEnumerable<TRecord> records, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(records);
        TRecord[] batch = records.ToArray();
        if (batch.Length == 0)
        {
            return;
        }

        float[][] vectors = new float[batch.Length][];
        string[] keyJsons = new string[batch.Length];
        string[] recordJsons = new string[batch.Length];
        var seenKeys = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < batch.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            TRecord record = batch[i] ?? throw new ArgumentException("The record batch cannot contain null entries.", nameof(records));
            object? key = _model.KeyProperty.GetValueAsObject(record);
            if (key is null || IsDefaultKey(key) && _model.KeyProperty.IsAutoGenerated)
            {
                key = GenerateKey();
                _model.KeyProperty.SetValueAsObject(record, key);
            }

            keyJsons[i] = SerializeKeyObject(key);
            if (!seenKeys.Add(keyJsons[i]))
            {
                throw new ArgumentException("The record batch contains duplicate keys.", nameof(records));
            }

            vectors[i] = PrepareStoredVector(GetVector(
                _model.VectorProperty.GetValueAsObject(record),
                _model.VectorProperty.ModelName,
                _model.VectorProperty.Dimensions));
            recordJsons[i] = JsonSerializer.Serialize(record, _serializerOptions);
        }

        int dimension = vectors[0].Length;
        var packedInput = new float[checked(batch.Length * dimension)];
        for (int i = 0; i < vectors.Length; i++)
        {
            if (vectors[i].Length != dimension)
            {
                throw new ArgumentException("All vectors in an upsert batch must have the same dimension.", nameof(records));
            }

            vectors[i].CopyTo(packedInput, i * dimension);
        }

        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            State state = await LoadIfExistsAsync(cancellationToken).ConfigureAwait(false) ?? NewState();
            ulong[] ids = new ulong[batch.Length];
            for (int i = 0; i < batch.Length; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                string keyJson = keyJsons[i];
                ulong id;
                if (state.KeyToId.TryGetValue(keyJson, out id))
                {
                    state.Index.Remove(id);
                    state.Records.Remove(IdKey(id));
                }
                else
                {
                    if (state.NextId == ulong.MaxValue)
                    {
                        throw new InvalidOperationException("The collection has exhausted its internal record identifiers.");
                    }

                    id = state.NextId++;
                    state.KeyToId.Add(keyJson, id);
                }

                ids[i] = id;
                state.Records[IdKey(id)] = recordJsons[i];
            }

            state.Index.AddWithIds2D(packedInput, dimension, ids);
            await SaveAsync(state, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _gate.Release();
        }
    }

    public override async IAsyncEnumerable<VectorSearchResult<TRecord>> SearchAsync<TInput>(
        TInput searchValue,
        int top,
        VectorSearchOptions<TRecord>? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(top);
        if (top == 0)
        {
            yield break;
        }

        List<VectorSearchResult<TRecord>> results = [];
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            State? state = await LoadIfExistsAsync(cancellationToken).ConfigureAwait(false);
            if (state is null || state.Records.Count == 0)
            {
                yield break;
            }

            VectorPropertyModel vectorProperty = _model.GetVectorPropertyOrSingle(options ?? new VectorSearchOptions<TRecord>());
            float[] query = PrepareQueryVector(GetVector(searchValue, nameof(searchValue), vectorProperty.Dimensions));
            Func<TRecord, bool>? predicate = options?.Filter?.Compile();
            int skip = options?.Skip ?? 0;
            ArgumentOutOfRangeException.ThrowIfNegative(skip);
            ulong[]? allowlist = null;
            if (predicate is not null)
            {
                var ids = new List<ulong>(state.Records.Count);
                foreach ((string idText, string json) in state.Records)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    if (predicate(DeserializeRecord(json)))
                    {
                        ids.Add(ulong.Parse(idText, CultureInfo.InvariantCulture));
                    }
                }

                if (ids.Count == 0)
                {
                    yield break;
                }

                allowlist = [.. ids];
            }

            int candidateCount = options?.ScoreThreshold is not null
                ? (allowlist?.Length ?? state.Records.Count)
                : checked(top + skip);
            (float[] scores, ulong[] resultIds) = state.Index.SearchWithAllowlist(query, candidateCount, allowlist);
            double? scoreThreshold = options?.ScoreThreshold;
            for (int i = 0; i < resultIds.Length; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                float score = _distanceFunction switch
                {
                    DistanceFunction.CosineDistance => 1f - scores[i],
                    DistanceFunction.NegativeDotProductSimilarity => -scores[i],
                    _ => scores[i]
                };
                bool isDistance = _distanceFunction is DistanceFunction.CosineDistance or DistanceFunction.NegativeDotProductSimilarity;
                if (scoreThreshold is double threshold &&
                    (isDistance ? score > threshold : score < threshold))
                {
                    continue;
                }

                if (skip > 0)
                {
                    skip--;
                    continue;
                }

                if (!state.Records.TryGetValue(IdKey(resultIds[i]), out string? json))
                {
                    throw new InvalidDataException($"Index contains record id {resultIds[i]} with no corresponding record payload.");
                }

                TRecord record = DeserializeRecord(json);
                if (options?.IncludeVectors != true)
                {
                    record = WithoutVectors(record);
                }

                results.Add(new VectorSearchResult<TRecord>(record, score));
            }
        }
        finally
        {
            _gate.Release();
        }

        foreach (VectorSearchResult<TRecord> result in results)
        {
            cancellationToken.ThrowIfCancellationRequested();
            yield return result;
        }
    }

    public override object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        return serviceType.IsInstanceOfType(this) ? this : null;
    }

    private async Task<State?> LoadIfExistsAsync(CancellationToken cancellationToken)
        => File.Exists(_path) ? await LoadAsync(cancellationToken).ConfigureAwait(false) : null;

    private async Task<State> LoadAsync(CancellationToken cancellationToken)
    {
        CollectionImage image = await CollectionStorage.ReadImageAsync(_path, cancellationToken).ConfigureAwait(false);
        if (image.Index is null || image.KeyToId is null || image.Records is null || image.Schema is null || image.Name is null)
        {
            throw new InvalidDataException($"Collection image '{_path}' is missing required data.");
        }

        if (!string.Equals(image.Name, Name, StringComparison.Ordinal))
        {
            throw new InvalidDataException($"Collection image '{_path}' is named '{image.Name}', not '{Name}'.");
        }

        if (!string.Equals(image.Schema, _schema, StringComparison.Ordinal))
        {
            throw new InvalidOperationException($"Collection '{Name}' was created with a different record schema.");
        }

        IdMapIndex index = IdMapIndex.Load(image.Index);
        if (image.NextId == 0 || image.KeyToId.Count != image.Records.Count || index.Len != image.Records.Count ||
            image.KeyToId.Values.Distinct().Count() != image.KeyToId.Count ||
            image.KeyToId.Values.Any(id => id >= image.NextId) ||
            image.KeyToId.Values.Any(id => !index.Contains(id) || !image.Records.ContainsKey(IdKey(id))) ||
            image.Records.Keys.Any(id => !ulong.TryParse(id, CultureInfo.InvariantCulture, out ulong value) || !index.Contains(value)))
        {
            throw new InvalidDataException($"Collection image '{_path}' has inconsistent record and index mappings.");
        }

        return new State(index, image.KeyToId, image.Records, image.NextId);
    }

    private State NewState()
        => new(IdMapIndex.NewLazy(_bitWidth), new Dictionary<string, ulong>(StringComparer.Ordinal),
            new Dictionary<string, string>(StringComparer.Ordinal), 1);

    private async Task SaveAsync(State state, CancellationToken cancellationToken)
        => await CollectionStorage.WriteImageAsync(_path, new CollectionImage
        {
            Name = Name,
            Schema = _schema,
            Index = state.Index.ToBytes(),
            KeyToId = state.KeyToId,
            Records = state.Records,
            NextId = state.NextId
        }, cancellationToken).ConfigureAwait(false);

    private string SerializeKey(TKey key)
        => SerializeKeyObject(key);

    private string SerializeKeyObject(object key)
        => JsonSerializer.Serialize(key, key.GetType(), _serializerOptions);

    private string IdKey(ulong id) => id.ToString(CultureInfo.InvariantCulture);

    private TRecord DeserializeRecord(string json)
    {
        TRecord record = JsonSerializer.Deserialize<TRecord>(json, _serializerOptions)
            ?? throw new InvalidDataException($"A record in collection '{Name}' deserialized to null.");

        if (record is IDictionary<string, object?> dynamicRecord)
        {
            foreach (PropertyModel property in _model.Properties)
            {
                if (dynamicRecord.TryGetValue(property.ModelName, out object? value) && value is JsonElement element)
                {
                    dynamicRecord[property.ModelName] = element.Deserialize(property.Type, _serializerOptions);
                }
            }
        }

        return record;
    }

    private TRecord WithoutVectors(TRecord record)
    {
        if (_model.VectorProperties.Count == 0)
        {
            return record;
        }

        JsonNode? node = JsonSerializer.SerializeToNode(record, _serializerOptions);
        if (node is JsonObject obj)
        {
            foreach (VectorPropertyModel property in _model.VectorProperties)
            {
                string jsonName;
                if (property.PropertyInfo is null)
                {
                    jsonName = _serializerOptions.DictionaryKeyPolicy?.ConvertName(property.ModelName) ?? property.ModelName;
                }
                else
                {
                    jsonName = property.PropertyInfo.GetCustomAttributes(typeof(JsonPropertyNameAttribute), inherit: true)
                        .OfType<JsonPropertyNameAttribute>()
                        .FirstOrDefault()?.Name
                        ?? _serializerOptions.PropertyNamingPolicy?.ConvertName(property.PropertyInfo.Name)
                        ?? property.PropertyInfo.Name;
                }

                obj.Remove(jsonName);
            }
        }

        return node is null
            ? record
            : node.Deserialize<TRecord>(_serializerOptions) ?? record;
    }

    private static float[] GetVector(object? value, string propertyName, int? dimensions)
    {
        float[] vector = value switch
        {
            float[] array => array,
            Memory<float> memory => memory.ToArray(),
            ReadOnlyMemory<float> memory => memory.ToArray(),
            IEnumerable<float> enumerable => enumerable.ToArray(),
            _ => throw new ArgumentException($"Vector property '{propertyName}' must contain float coordinates.")
        };

        if (vector.Length == 0)
        {
            throw new ArgumentException($"Vector property '{propertyName}' cannot be empty.");
        }

        foreach (float coordinate in vector)
        {
            if (!float.IsFinite(coordinate) || MathF.Abs(coordinate) >= 1e16f)
            {
                throw new ArgumentException($"Vector property '{propertyName}' contains an invalid coordinate.");
            }
        }

        if (dimensions is int expected && vector.Length != expected)
        {
            throw new ArgumentException($"Vector property '{propertyName}' requires {expected} dimensions, but received {vector.Length}.");
        }

        return vector;
    }

    private float[] PrepareStoredVector(float[] vector)
        => _distanceFunction is DistanceFunction.CosineSimilarity or DistanceFunction.CosineDistance
            ? Normalize(vector)
            : vector;

    private float[] PrepareQueryVector(float[] vector)
        => _distanceFunction is DistanceFunction.CosineSimilarity or DistanceFunction.CosineDistance
            ? Normalize(vector)
            : vector;

    private static float[] Normalize(float[] vector)
    {
        float norm = TensorPrimitives.Norm(vector);
        if (norm > 1e-10f)
        {
            TensorPrimitives.Divide(vector, norm, vector);
        }
        else
        {
            Array.Clear(vector);
        }

        return vector;
    }

    private static object GenerateKey()
    {
        Type keyType = typeof(TKey);
        if (keyType == typeof(Guid))
        {
            return Guid.NewGuid();
        }

        throw new InvalidOperationException($"Automatic key generation is not supported for key type '{keyType}'.");
    }

    private static bool IsDefaultKey(object key)
        => key switch
        {
            Guid guid => guid == Guid.Empty,
            string text => string.IsNullOrEmpty(text),
            _ => key.GetType().IsValueType && Equals(key, Activator.CreateInstance(key.GetType()))
        };

    private static string CreateSchemaFingerprint(CollectionModel model)
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
            entries.Add($"{kind}|{property.ModelName}|{property.StorageName}|{property.Type.AssemblyQualifiedName}");
        }

        entries.Sort(StringComparer.Ordinal);
        return string.Join('\n', entries);
    }

    private static IEnumerable<TRecord> ApplyOrdering(
        IEnumerable<TRecord> records,
        Func<FilteredRecordRetrievalOptions<TRecord>.OrderByDefinition, FilteredRecordRetrievalOptions<TRecord>.OrderByDefinition>? orderByFactory)
    {
        IOrderedEnumerable<TRecord>? ordered = null;
        if (orderByFactory is null)
        {
            return records;
        }

        var orderBy = orderByFactory(new FilteredRecordRetrievalOptions<TRecord>.OrderByDefinition());
        foreach (FilteredRecordRetrievalOptions<TRecord>.OrderByDefinition.SortInfo sort in orderBy.Values)
        {
            Func<TRecord, object?> selector = sort.PropertySelector.Compile();
            ordered = ordered is null
                ? sort.Ascending
                    ? records.OrderBy(selector, Comparer<object?>.Default)
                    : records.OrderByDescending(selector, Comparer<object?>.Default)
                : sort.Ascending
                    ? ordered.ThenBy(selector, Comparer<object?>.Default)
                    : ordered.ThenByDescending(selector, Comparer<object?>.Default);
        }

        return ordered ?? records;
    }

    private sealed class State(IdMapIndex index, Dictionary<string, ulong> keyToId, Dictionary<string, string> records, ulong nextId)
    {
        public IdMapIndex Index { get; } = index;
        public Dictionary<string, ulong> KeyToId { get; } = keyToId;
        public Dictionary<string, string> Records { get; } = records;
        public ulong NextId { get; set; } = nextId;
    }
}
