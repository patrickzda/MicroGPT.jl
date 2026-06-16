using Test
using MicroGPT

@testset "Tokenizer.jl" begin
    # Fixed input used across most testsets.
    # unique sorted chars of "hello"+"world" = ['d','e','h','l','o','r','w']
    # Token IDs are 1-based: 'd'->1 ... 'w'->7, BOS = 8, vocab_size = 8
    docs = ["hello", "world"]
    tok = Tokenizer(docs)
    BOS = tok.bos

    @testset "Vocabulary" begin
        @testset "uchars is sorted" begin
            @test tok.uchars == sort(tok.uchars)
        end

        @testset "uchars deduplicated" begin
            # Characters appearing in multiple docs should only occur once
            @test length(tok.uchars) == length(unique(tok.uchars))
            @test tok.uchars == ['d', 'e', 'h', 'l', 'o', 'r', 'w']
        end

        @testset "BOS value" begin
            @test BOS == length(tok.uchars) + 1
            @test BOS == 8
        end

        @testset "vocab size" begin
            @test tok.vocab_size == length(tok.uchars) + 1
            @test tok.vocab_size == 8
        end

        @testset "char2id matches uchars indices" begin
            for (i, c) in enumerate(tok.uchars)
                @test tok.char2id[c] == i
            end
        end
    end

    @testset "Encode" begin
        @testset "BOS wraps the sequence" begin
            ids = encode(tok, "do")
            # Every encoded sequence starts and ends with BOS
            @test first(ids) == BOS
            @test last(ids)  == BOS
        end

        @testset "token IDs" begin
            # 'd' -> uchars[1] -> token ID 1
            # 'o' -> uchars[5] -> token ID 5
            @test encode(tok, "do") == [BOS, 1, 5, BOS]
        end

        @testset "output length" begin
            @test length(encode(tok, "hello")) == length("hello") + 2
        end

        @testset "empty string" begin
            @test encode(tok, "") == [BOS, BOS]
        end
    end

    @testset "Decode" begin
        @testset "BOS tokens are stripped" begin
            # BOS markers should not appear in decoded text
            @test decode(tok, [BOS])      == ""
            @test decode(tok, [BOS, BOS]) == ""
        end

        @testset "string reconstruction" begin
            # token IDs [1, 5] -> ['d', 'o']
            @test decode(tok, [BOS, 1, 5, BOS]) == "do"
        end
    end

    @testset "Round-trip" begin
        # Encoding then decoding should recover the original string
        @testset "normal strings" begin
            for doc in ["hello", "world", "he", "d", "orl"]
                @test decode(tok, encode(tok, doc)) == doc
            end
        end

        @testset "empty string" begin
            @test decode(tok, encode(tok, "")) == ""
        end
    end

    @testset "Edge cases" begin
        @testset "single-char input" begin
            t = Tokenizer(["aaa"])
            @test t.uchars == ['a']
            @test t.bos == 2          # BOS
            @test t.vocab_size == 2
            @test encode(t, "a") == [2, 1, 2]
            @test decode(t, encode(t, "a")) == "a"
        end

        @testset "empty input entry" begin
            # ["ab", ""] should produce the same vocab as ["ab"]
            t1 = Tokenizer(["ab"])
            t2 = Tokenizer(["ab", ""])
            @test t1.uchars == t2.uchars
            @test t1.bos == t2.bos
            @test t1.vocab_size == t2.vocab_size
        end

        @testset "multi-doc deduplication" begin
            t = Tokenizer(["aab", "bbc"])
            @test t.uchars == ['a', 'b', 'c']
        end
    end

    @testset "Error handling" begin
        @testset "unknown char" begin
            # Unknown characters cannot be mapped to token IDs
            t = Tokenizer(["ab"])
            @test_throws ArgumentError encode(t, "z")
        end

        @testset "unknown char mid-string" begin
            t = Tokenizer(["ab"])
            @test_throws ArgumentError encode(t, "abz")
        end

        @testset "empty vocab encode" begin
            t = Tokenizer(String[])
            @test_throws ArgumentError encode(t, "a")
        end

        @testset "decode out-of-range id" begin
            # IDs above the vocabulary (but not the BOS token) are invalid
            t = Tokenizer(["ab"])           # uchars = ['a','b'], bos = 3
            @test_throws ArgumentError decode(t, [99])
        end

        @testset "empty input empty string" begin
            # Special case: no vocabulary, only BOS token exists
            t = Tokenizer(String[])
            @test t.bos == 1
            @test encode(t, "") == [1, 1]
            @test decode(t, encode(t, "")) == ""
        end
    end
end