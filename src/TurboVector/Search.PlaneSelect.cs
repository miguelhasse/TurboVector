namespace TurboVector;

public static partial class Search
{
    internal readonly record struct PlaneCandidate(float Score, int Index);

    // Rust's score-only unstable selection affects which tied candidates reach exact rescoring.
    // This follows the Rust standard-library introselect implementation used by the reference.
    internal static void SelectPlaneCandidate(Span<PlaneCandidate> values, int index, bool tieByIndex)
    {
        if (index == 0 || index == values.Length - 1)
        {
            SelectExtreme(values, index, tieByIndex);
            return;
        }

        var ancestor = default(PlaneCandidate?);
        var limit = 16;
        while (true)
        {
            if (values.Length <= 16)
            {
                InsertionSort(values, tieByIndex);
                return;
            }
            if (limit-- == 0)
            {
                MedianOfMedians(values, index, tieByIndex);
                return;
            }

            var pivot = ChoosePivot(values, tieByIndex);
            if (ancestor is { } previous && !CandidateLess(previous, values[pivot], tieByIndex))
            {
                var middle = PartitionCandidates(values, pivot, tieByIndex, equalFirst: true) + 1;
                if (middle > index) return;
                values = values[middle..];
                index -= middle;
                ancestor = null;
                continue;
            }

            var position = PartitionCandidates(values, pivot, tieByIndex, equalFirst: false);
            if (position == index) return;
            if (position < index)
            {
                ancestor = values[position];
                values = values[(position + 1)..];
                index -= position + 1;
            }
            else
            {
                values = values[..position];
            }
        }
    }

    private static bool CandidateLess(PlaneCandidate left, PlaneCandidate right, bool tieByIndex)
    {
        if (left.Score > right.Score) return true;
        if (left.Score < right.Score) return false;
        return tieByIndex && left.Index < right.Index;
    }

    private static void SelectExtreme(Span<PlaneCandidate> values, int index, bool tieByIndex)
    {
        var selected = 0;
        for (var i = 1; i < values.Length; i++)
        {
            var replace = index == 0
                ? CandidateLess(values[i], values[selected], tieByIndex)
                : CandidateLess(values[selected], values[i], tieByIndex);
            if (replace) selected = i;
        }
        (values[selected], values[index]) = (values[index], values[selected]);
    }

    private static void InsertionSort(Span<PlaneCandidate> values, bool tieByIndex)
    {
        for (var i = 1; i < values.Length; i++)
        {
            var value = values[i];
            var j = i;
            while (j > 0 && CandidateLess(value, values[j - 1], tieByIndex))
            {
                values[j] = values[j - 1];
                j--;
            }
            values[j] = value;
        }
    }

    private static int ChoosePivot(ReadOnlySpan<PlaneCandidate> values, bool tieByIndex)
    {
        var eighth = values.Length / 8;
        return values.Length < 64
            ? MedianOfThree(values, 0, 4 * eighth, 7 * eighth, tieByIndex)
            : PseudoMedian(values, 0, 4 * eighth, 7 * eighth, eighth, tieByIndex);
    }

    private static int PseudoMedian(ReadOnlySpan<PlaneCandidate> values, int a, int b, int c, int length, bool tieByIndex)
    {
        if (length * 8 >= 64)
        {
            var eighth = length / 8;
            a = PseudoMedian(values, a, a + 4 * eighth, a + 7 * eighth, eighth, tieByIndex);
            b = PseudoMedian(values, b, b + 4 * eighth, b + 7 * eighth, eighth, tieByIndex);
            c = PseudoMedian(values, c, c + 4 * eighth, c + 7 * eighth, eighth, tieByIndex);
        }
        return MedianOfThree(values, a, b, c, tieByIndex);
    }

    private static int MedianOfThree(ReadOnlySpan<PlaneCandidate> values, int a, int b, int c, bool tieByIndex)
    {
        var x = CandidateLess(values[a], values[b], tieByIndex);
        var y = CandidateLess(values[a], values[c], tieByIndex);
        if (x != y) return a;
        return CandidateLess(values[b], values[c], tieByIndex) ^ x ? c : b;
    }

    private static int PartitionCandidates(Span<PlaneCandidate> values, int pivot, bool tieByIndex, bool equalFirst)
    {
        (values[0], values[pivot]) = (values[pivot], values[0]);
        var pivotValue = values[0];
        var tail = values[1..];
        if (tail.IsEmpty) return 0;

        var saved = tail[0];
        var gap = 0;
        var less = 0;
        for (var right = 1; right <= tail.Length; right++)
        {
            var value = right == tail.Length ? saved : tail[right];
            var isLess = equalFirst
                ? !CandidateLess(pivotValue, value, tieByIndex)
                : CandidateLess(value, pivotValue, tieByIndex);
            tail[gap] = tail[less];
            tail[less] = value;
            gap = right;
            if (isLess) less++;
        }

        (values[0], values[less]) = (values[less], values[0]);
        return less;
    }

    private static void MedianOfMedians(Span<PlaneCandidate> values, int index, bool tieByIndex)
    {
        while (true)
        {
            if (values.Length <= 16)
            {
                InsertionSort(values, tieByIndex);
                return;
            }
            if (index == 0 || index == values.Length - 1)
            {
                SelectExtreme(values, index, tieByIndex);
                return;
            }

            var fraction = values.Length <= 1024 ? values.Length / 12
                : values.Length <= 128 * 1024 ? values.Length / 64 : values.Length / 1024;
            var middle = fraction / 2;
            var lo = values.Length / 2 - middle;
            var hi = lo + fraction;
            var gap = (values.Length - 9 * fraction) / 4;
            var a = lo - 4 * fraction - gap;
            var b = hi + gap;
            for (var i = lo; i < hi; i++)
            {
                Ninther(values, a, i - fraction, b, a + 1, i, b + 1, a + 2, i + fraction, b + 2, tieByIndex);
                a += 3;
                b += 3;
            }
            MedianOfMedians(values.Slice(lo, fraction), middle, tieByIndex);
            var position = PartitionCandidates(values, lo + middle, tieByIndex, equalFirst: false);
            if (position == index) return;
            if (position > index)
            {
                values = values[..position];
            }
            else
            {
                values = values[(position + 1)..];
                index -= position + 1;
            }
        }
    }

    private static void Ninther(Span<PlaneCandidate> values, int a, int b, int c, int d, int e, int f, int g, int h, int i, bool tieByIndex)
    {
        b = MedianIndex(values, a, b, c, tieByIndex);
        h = MedianIndex(values, g, h, i, tieByIndex);
        if (CandidateLess(values[h], values[b], tieByIndex)) (b, h) = (h, b);
        if (CandidateLess(values[f], values[d], tieByIndex)) (d, f) = (f, d);
        if (!CandidateLess(values[e], values[d], tieByIndex))
        {
            if (CandidateLess(values[f], values[e], tieByIndex))
            {
                d = f;
            }
            else
            {
                if (CandidateLess(values[e], values[b], tieByIndex)) (values[e], values[b]) = (values[b], values[e]);
                else if (CandidateLess(values[h], values[e], tieByIndex)) (values[e], values[h]) = (values[h], values[e]);
                return;
            }
        }
        if (CandidateLess(values[d], values[b], tieByIndex)) d = b;
        else if (CandidateLess(values[h], values[d], tieByIndex)) d = h;
        (values[d], values[e]) = (values[e], values[d]);
    }

    private static int MedianIndex(ReadOnlySpan<PlaneCandidate> values, int a, int b, int c, bool tieByIndex)
    {
        if (CandidateLess(values[c], values[a], tieByIndex)) (a, c) = (c, a);
        if (CandidateLess(values[c], values[b], tieByIndex)) return c;
        return CandidateLess(values[b], values[a], tieByIndex) ? a : b;
    }
}
