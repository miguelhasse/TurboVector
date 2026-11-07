namespace TurboVector.Tests;

public class PlaneSelectionParityTests
{
    [Theory]
    [InlineData(2, "08cd4c29d1e47d34 08cd4c29d1e47d34 08cd4c29d1e47d34 08cd4c29d1e47d34 08cd4c29d1e47d34 08cd4c29d1e47d34 08cd4c29d1e47d34 08cd4c29d1e47d34")]
    [InlineData(15, "47ebffbe9d2dc21a b5d60a74019cf5ea b5d60a74019cf5ea 947101ae51a5ea0a 47ebffbe9d2dc21a b5d60a74019cf5ea b5d60a74019cf5ea 947101ae51a5ea0a")]
    [InlineData(16, "4315ecb8ddcb59e5 313782249b715795 313782249b715795 cfc655387e8c2f25 4315ecb8ddcb59e5 313782249b715795 313782249b715795 cfc655387e8c2f25")]
    [InlineData(17, "c9187e90e3f6dfc5 87e4f32ebb230505 5728fb020424a395 e2575053967fb4b5 c9187e90e3f6dfc5 73a9f8db5c62c7d5 0b44ed3c72c17b85 17187757633ac705")]
    [InlineData(31, "e332752c9481ec8a dc5f7f26f111efea a56fd70b4e11e9ba 939ac108edc5639a e332752c9481ec8a 997668cc48260dea 0108b227e9f4421a eb82f719fdeda40a")]
    [InlineData(32, "7d450057902d6c65 8d7c19631ea8fc85 7cbdb4f5fd98ecb5 5108ad2ffd4907a5 7d450057902d6c65 e5ab33664fa39025 cda529c254efbba5 ad49ca8620d23515")]
    [InlineData(63, "6577da61a392902a 769d9306a502de1a 56ccb3e47bd920fa 9b8f2bd3fd3ff0fa 6577da61a392902a 1f5cc8d48c26fcea 96bb0344a7a2eb5a 27052fafda500a3a")]
    [InlineData(64, "d5511bd42a2c07e5 ceae20030ac9e245 8df7a5ca8bc36665 e940b6ba38425ce5 d5511bd42a2c07e5 5e4745f1693e82f5 914353cdfcc02e55 633758989b092565")]
    [InlineData(127, "7d3d402d1fd6a46a 5174d87d1260ecba ae493bd75557c35a 35b3b03cec1c39ba 7d3d402d1fd6a46a bb26d6c97405e6fa 2f2e8ddde62887fa 28c644efd46d03aa")]
    [InlineData(256, "7bf07488960e87e5 b94dd86dfa64f285 b80d6a59876cc3f5 779a0733194ec265 7bf07488960e87e5 29a4494af4cbb885 f9f27e4c7cff1265 fadb5a42ecf73b95")]
    [InlineData(768, "3496fd8ed9f0fae5 c26eec2c2f3b0bb5 e5357386ba12fa61 719d3764e0033f99 3496fd8ed9f0fae5 87cde70034d116e9 fd89ec56063c1a81 1596f5c1ce2f3065")]
    [InlineData(1280, "f1ffa40714b360e5 5095bae267c32529 c5146505235497cd d07c1d96e4cdeb1d f1ffa40714b360e5 75c1795bbbef290d 075e6ed208aede75 9021930b099ba4dd")]
    [InlineData(4096, "2c87d84d8fbe39e5 d47215ed571751d9 b543f7c1ebf37b09 beb6e7a6cc6cd765 2c87d84d8fbe39e5 cdc0d20e4a6c9249 cb948d5fc977b5d9 0a539881f2810a8d")]
    public void SelectionMatchesRustWithRepeatedScores(int length, string expected)
    {
        var hashes = expected.Split(' ');
        var ranks = new[] { 0, 1, length / 2, length - 1 };
        for (var mode = 0; mode < 2; mode++)
        {
            for (var r = 0; r < ranks.Length; r++)
            {
                var candidates = new Search.PlaneCandidate[length];
                var state = 0xc0ffeeUL;
                for (var i = 0; i < length; i++)
                {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    candidates[i] = new Search.PlaneCandidate((state >> 32) & 7, i);
                }

                Search.SelectPlaneCandidate(candidates, ranks[r], mode != 0);
                var hash = 0xcbf29ce484222325UL;
                foreach (var candidate in candidates)
                {
                    for (var b = 0; b < sizeof(int); b++)
                    {
                        hash ^= (byte)(candidate.Index >> (b * 8));
                        hash = unchecked(hash * 0x100000001b3UL);
                    }
                }
                Assert.Equal(hashes[mode * ranks.Length + r], hash.ToString("x16"));
            }
        }
    }
}
