using Test
using MicroGPT

@testset "Tokenizer.jl" begin
    # Fixed input used across most testsets.
    # unique sorted chars of "hello"+"world" = ['d','e','h','l','o','r','w']
    # BOS = 7, vocab_size = 8
    docs = ["hello", "world"]
    uchars, BOS, vocab_size, encode, decode = tokenizer_JuML(docs)

    @testset "Vocabulary" begin
        @testset "uchars is sorted" begin
            @test uchars == sort(uchars)
        end

        @testset "uchars deduplicated" begin
            # Characters appearing in multiple docs should only occur once
            @test length(uchars) == length(unique(uchars))
            @test uchars == ['d', 'e', 'h', 'l', 'o', 'r', 'w']
        end

        @testset "BOS value" begin
            @test BOS == length(uchars)
            @test BOS == 7
        end

        @testset "vocab size" begin
            @test vocab_size == length(uchars) + 1
            @test vocab_size == 8
        end
    end

    @testset "Encode" begin
        @testset "BOS wraps the sequence" begin
            ids = encode("do")
            # Every encoded sequence starts and ends with BOS
            @test first(ids) == BOS
            @test last(ids)  == BOS
        end

        @testset "token IDs" begin
            # 'd' -> uchars[1] -> token ID 0
            # 'o' -> uchars[5] -> token ID 4
            @test encode("do") == [BOS, 0, 4, BOS]
        end

        @testset "output length" begin
            @test length(encode("hello")) == length("hello") + 2
        end

        @testset "empty string" begin
            @test encode("") == [BOS, BOS]
        end
    end
        # BOS = 0, inner loop never runs
    @testset "Decode" begin
        @testset "BOS tokens are stripped" begin
            # BOS markers should not appear in decoded text
            @test decode([BOS])      == ""
            @test decode([BOS, BOS]) == ""
        end

        @testset "string reconstruction" begin
            # token IDs [0, 4] -> ['d', 'o']
            @test decode([BOS, 0, 4, BOS]) == "do"
        end
    end

    @testset "Round-trip" begin
        # Encoding then decoding should recover the original string
        @testset "normal strings" begin
            for doc in ["hello", "world", "he", "d", "orl"]
                @test decode(encode(doc)) == doc
            end
        end

        @testset "empty string" begin
            @test decode(encode("")) == ""
        end
    end

    @testset "Edge cases" begin
        @testset "single-char input" begin
            u, b, v, enc, dec = tokenizer_JuML(["aaa"])
            @test u == ['a']
            @test b == 1    # BOS
            @test v == 2    # vocab_size
            @test enc("a") == [1, 0, 1]
            @test dec(enc("a")) == "a"
        end

        @testset "empty input entry" begin
            # ["ab", ""] should produce the same vocab as ["ab"]
            u1, b1, v1, _, _ = tokenizer_JuML(["ab"])
            u2, b2, v2, _, _ = tokenizer_JuML(["ab", ""])
            @test u1 == u2
            @test b1 == b2
            @test v1 == v2
        end

        @testset "multi-doc deduplication" begin
            u, _, _, _, _ = tokenizer_JuML(["aab", "bbc"])
            @test u == ['a', 'b', 'c']
        end
    end

    @testset "Error handling" begin
        @testset "unknown char" begin
            # Unknown characters cannot be mapped to token IDs
            _, _, _, enc, _ = tokenizer_JuML(["ab"])
            @test_throws MethodError enc("z")
        end

        @testset "unknown char mid-string" begin
            _, _, _, enc, _ = tokenizer_JuML(["ab"])
            @test_throws MethodError enc("abz")
        end

        @testset "empty inout encode" begin
            _, _, _, enc, _ = tokenizer_JuML(String[])
            @test_throws MethodError enc("a")
        end

        @testset "empty input empty string" begin
            # Special case: no vocabulary, only BOS token exists
            _, b, _, enc, dec = tokenizer_JuML(String[])
            @test b == 0
            @test enc("") == [0, 0]
            @test dec(enc("")) == ""
        end
    end
end
