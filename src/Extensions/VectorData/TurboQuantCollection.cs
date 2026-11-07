using Microsoft.Extensions.VectorData;
using Microsoft.Extensions.VectorData.ProviderServices;
using System.Globalization;
using System.Linq.Expressions;
using System.Runtime.CompilerServices;
using TurboVector;

namespace Hasseware.Extensions.VectorData.TurboQuant;

/// <summary>A TurboQuant collection persisted through a <see cref="TurboQuantVectorStore"/>.</summary>
public sealed class TurboQuantCollection<TKey, TRecord> : VectorStoreCollection<TKey, TRecord>
    where TKey : notnull
    where TRecord : class
{
    private readonly TurboQuantVectorStore _store;
    private readonly RecordAccessor<TKey, TRecord> _accessor;
    private readonly CollectionSchema _schema;
    private readonly SemaphoreSlim _gate;

    internal TurboQuantCollection(TurboQuantVectorStore store, string name, RecordAccessor<TKey, TRecord> accessor)
    {
        _store = store;
        _accessor = accessor;
        _schema = accessor.Schema;
        Name = name;
        _gate = store.GetLock(name);
    }

    /// <inheritdoc />
    public override string Name { get; }

    /// <inheritdoc />
    public override async Task<bool> CollectionExistsAsync(CancellationToken cancellationToken = default)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            return await _store.ImageExistsAsync(Name, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <inheritdoc />
    public override async Task EnsureCollectionExistsAsync(CancellationToken cancellationToken = default)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (await LoadIfExistsAsync(cancellationToken).ConfigureAwait(false) is null)
            {
                await SaveAsync(NewState(), cancellationToken).ConfigureAwait(false);
            }
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <inheritdoc />
    public override async Task EnsureCollectionDeletedAsync(CancellationToken cancellationToken = default)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            await _store.DeleteImageAsync(Name, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <inheritdoc />
    public override async Task<TRecord?> GetAsync(
        TKey key,
        RecordRetrievalOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(key);
        string keyJson = _accessor.SerializeKey(key);
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            State? state = await LoadIfExistsAsync(cancellationToken).ConfigureAwait(false);
            if (state is null || !state.KeyToId.TryGetValue(keyJson, out ulong id) ||
                !state.Records.TryGetValue(IdKey(id), out string? json))
            {
                return default;
            }

            return _accessor.Deserialize(json, options?.IncludeVectors == true);
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <inheritdoc />
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

        var requestedSkip = options?.Skip ?? 0;
        ArgumentOutOfRangeException.ThrowIfNegative(requestedSkip);
        var includeVectors = options?.IncludeVectors == true;
        var predicate = filter.Compile();
        var result = new List<(TRecord? Record, string Json)>();
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var state = await LoadIfExistsAsync(cancellationToken).ConfigureAwait(false);
            if (state is null)
            {
                yield break;
            }

            foreach (var json in state.Records.Values)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var record = _accessor.Deserialize(json, includeVectors: true);
                if (predicate(record))
                {
                    result.Add((includeVectors ? record : null, json));
                }
            }
        }
        finally
        {
            _gate.Release();
        }

        var ordered = ApplyOrdering(result, options?.OrderBy, cancellationToken);
        foreach (var match in ordered.Skip(requestedSkip).Take(top))
        {
            cancellationToken.ThrowIfCancellationRequested();
            yield return match.Record ?? _accessor.Deserialize(match.Json, includeVectors: false);
        }
    }

    /// <inheritdoc />
    public override async Task DeleteAsync(TKey key, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(key);
        string keyJson = _accessor.SerializeKey(key);
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            State? state = await LoadIfExistsAsync(cancellationToken).ConfigureAwait(false);
            if (state is null || !state.KeyToId.Remove(keyJson, out ulong id))
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

    /// <inheritdoc />
    public override Task UpsertAsync(TRecord record, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(record);
        return UpsertAsync([record], cancellationToken);
    }

    /// <inheritdoc />
    public override async Task UpsertAsync(IEnumerable<TRecord> records, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(records);
        TRecord[] batch = records.ToArray();
        if (batch.Length == 0)
        {
            return;
        }

        VectorPropertyModel vectorProperty = _schema.VectorProperty;
        var vectors = new ReadOnlyMemory<float>[batch.Length];
        string[] keyJsons = new string[batch.Length];
        string[] recordJsons = new string[batch.Length];
        var seenKeys = new HashSet<string>(batch.Length, StringComparer.Ordinal);
        for (int i = 0; i < batch.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            TRecord record = batch[i] ?? throw new ArgumentException("The record batch cannot contain null entries.", nameof(records));
            keyJsons[i] = _accessor.PrepareKey(record);
            if (!seenKeys.Add(keyJsons[i]))
            {
                throw new ArgumentException("The record batch contains duplicate keys.", nameof(records));
            }

            vectors[i] = _accessor.GetVector(record);
            VectorValues.Validate(vectors[i].Span, vectorProperty.ModelName, vectorProperty.Dimensions);
            recordJsons[i] = _accessor.Serialize(record);
        }

        int dimension = vectors[0].Length;
        var packedInput = new float[checked(batch.Length * dimension)];
        for (int i = 0; i < vectors.Length; i++)
        {
            if (vectors[i].Length != dimension)
            {
                throw new ArgumentException("All vectors in an upsert batch must have the same dimension.", nameof(records));
            }

            Span<float> destination = packedInput.AsSpan(i * dimension, dimension);
            vectors[i].Span.CopyTo(destination);
            if (_schema.NormalizesVectors)
            {
                VectorValues.Normalize(destination);
            }
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
                if (state.KeyToId.TryGetValue(keyJson, out ulong id))
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

    /// <inheritdoc />
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

        int skip = options?.Skip ?? 0;
        ArgumentOutOfRangeException.ThrowIfNegative(skip);
        VectorPropertyModel vectorProperty = options?.VectorProperty is null
            ? _schema.VectorProperty
            : _schema.Model.GetVectorPropertyOrSingle(options);
        ReadOnlyMemory<float> query = VectorValues.ToMemory(searchValue, nameof(searchValue));
        VectorValues.Validate(query.Span, nameof(searchValue), vectorProperty.Dimensions);
        if (_schema.NormalizesVectors)
        {
            float[] normalized = query.ToArray();
            VectorValues.Normalize(normalized);
            query = normalized;
        }

        Func<TRecord, bool>? predicate = options?.Filter?.Compile();
        bool includeVectors = options?.IncludeVectors == true;
        List<VectorSearchResult<TRecord>> results = [];
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            State? state = await LoadIfExistsAsync(cancellationToken).ConfigureAwait(false);
            if (state is null || state.Records.Count == 0)
            {
                yield break;
            }

            ulong[]? allowlist = null;
            if (predicate is not null)
            {
                var ids = new List<ulong>(state.Records.Count);
                foreach ((string idText, string json) in state.Records)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    if (predicate(_accessor.Deserialize(json, includeVectors: true)))
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

            int candidateCount = (int)Math.Min((long)top + skip, allowlist?.Length ?? state.Records.Count);
            (float[] scores, ulong[] resultIds) = state.Index.SearchWithAllowlist(query.Span, candidateCount, allowlist);
            double? scoreThreshold = options?.ScoreThreshold;
            bool isDistance = _schema.DistanceFunction is DistanceFunction.CosineDistance or DistanceFunction.NegativeDotProductSimilarity;
            for (int i = 0; i < resultIds.Length; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                float score = _schema.DistanceFunction switch
                {
                    DistanceFunction.CosineDistance => 1f - scores[i],
                    DistanceFunction.NegativeDotProductSimilarity => -scores[i],
                    _ => scores[i]
                };
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

                results.Add(new VectorSearchResult<TRecord>(_accessor.Deserialize(json, includeVectors), score));
                if (results.Count == top)
                {
                    break;
                }
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

    /// <inheritdoc />
    public override object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        return serviceType.IsInstanceOfType(this) ? this : null;
    }

    private async Task<State?> LoadIfExistsAsync(CancellationToken cancellationToken)
    {
        CollectionImage? image = await _store.ReadImageAsync(Name, cancellationToken).ConfigureAwait(false);
        if (image is null)
        {
            return null;
        }

        if (image.Index is null || image.KeyToId is null || image.Records is null || image.Schema is null || image.Name is null)
        {
            throw new InvalidDataException($"The image of collection '{Name}' is missing required data.");
        }

        if (!string.Equals(image.Name, Name, StringComparison.Ordinal))
        {
            throw new InvalidDataException($"The image stored for collection '{Name}' is named '{image.Name}'.");
        }

        if (!string.Equals(image.Schema, _schema.Fingerprint, StringComparison.Ordinal))
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
            throw new InvalidDataException($"The image of collection '{Name}' has inconsistent record and index mappings.");
        }

        return new State(index, image.KeyToId, image.Records, image.NextId);
    }

    private State NewState()
        => new(IdMapIndex.NewLazy(_store.BitWidth), new Dictionary<string, ulong>(StringComparer.Ordinal),
            new Dictionary<string, string>(StringComparer.Ordinal), 1);

    private ValueTask SaveAsync(State state, CancellationToken cancellationToken)
        => _store.WriteImageAsync(new CollectionImage
        {
            Name = Name,
            Schema = _schema.Fingerprint,
            Index = state.Index.ToBytes(),
            KeyToId = state.KeyToId,
            Records = state.Records,
            NextId = state.NextId
        }, cancellationToken);

    private static string IdKey(ulong id) => id.ToString(CultureInfo.InvariantCulture);

    private IEnumerable<(TRecord? Record, string Json)> ApplyOrdering(
        IReadOnlyList<(TRecord? Record, string Json)> records,
        Func<FilteredRecordRetrievalOptions<TRecord>.OrderByDefinition, FilteredRecordRetrievalOptions<TRecord>.OrderByDefinition>? orderByFactory,
        CancellationToken cancellationToken)
    {
        if (orderByFactory is null)
        {
            return records;
        }

        var sorts = orderByFactory(new FilteredRecordRetrievalOptions<TRecord>.OrderByDefinition()).Values;
        if (sorts.Count == 0)
        {
            return records;
        }

        var selectors = new Func<TRecord, object?>[sorts.Count];
        var keys = new object?[sorts.Count][];
        for (var i = 0; i < sorts.Count; i++)
        {
            selectors[i] = sorts[i].PropertySelector.Compile();
            keys[i] = new object?[records.Count];
        }

        // Keep sort keys, rather than every vector, alive until pagination is complete.
        for (var i = 0; i < records.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var record = records[i].Record ?? _accessor.Deserialize(records[i].Json, includeVectors: true);
            for (var sortIndex = 0; sortIndex < selectors.Length; sortIndex++)
            {
                keys[sortIndex][i] = selectors[sortIndex](record);
            }
        }

        var indices = Enumerable.Range(0, records.Count);
        var ordered = default(IOrderedEnumerable<int>);
        for (var i = 0; i < sorts.Count; i++)
        {
            var column = keys[i];
            var sort = sorts[i];
            ordered = ordered is null
                ? sort.Ascending
                    ? indices.OrderBy(index => column[index], Comparer<object?>.Default)
                    : indices.OrderByDescending(index => column[index], Comparer<object?>.Default)
                : sort.Ascending
                    ? ordered.ThenBy(index => column[index], Comparer<object?>.Default)
                    : ordered.ThenByDescending(index => column[index], Comparer<object?>.Default);
        }

        return (ordered ?? indices).Select(index => records[index]);
    }

    private sealed class State(IdMapIndex index, Dictionary<string, ulong> keyToId, Dictionary<string, string> records, ulong nextId)
    {
        public IdMapIndex Index { get; } = index;
        public Dictionary<string, ulong> KeyToId { get; } = keyToId;
        public Dictionary<string, string> Records { get; } = records;
        public ulong NextId { get; set; } = nextId;
    }
}
