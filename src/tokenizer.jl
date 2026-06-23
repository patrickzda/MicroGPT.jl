
"""
	Tokenizer

A character-level tokenizer built from a corpus of documents.

Fields:
- `uchars`: sorted, deduplicated vocabulary characters.
- `char2id`: reverse lookup from character to its token ID.
- `bos`: boundary token ID used to wrap every encoded sequence.
- `vocab_size`: number of distinct tokens, including the `bos` token.
"""
struct Tokenizer
	uchars::Vector{Char}
	char2id::Dict{Char,Int}
	bos::Int
	vocab_size::Int
end

"""
	Tokenizer(docs::AbstractVector{<:AbstractString})

Build a `Tokenizer` from `docs`. The vocabulary is the sorted set of all
characters appearing in the corpus.
"""
function Tokenizer(docs::AbstractVector{<:AbstractString})
	uchars = sort(unique(join(docs)))
	char2id = Dict(c => i for (i, c) in enumerate(uchars))
	bos = length(uchars) + 1
	vocab_size = length(uchars) + 1
	return Tokenizer(uchars, char2id, bos, vocab_size)
end

"""
	encode(tok::Tokenizer, doc::AbstractString) -> Vector{Int}

Encode `doc` into token IDs, wrapped with the boundary token at both ends.
Throws `ArgumentError` if `doc` contains a character outside the vocabulary.
"""
function encode(tok::Tokenizer, doc::AbstractString)
	token_ids = Vector{Int}()
	push!(token_ids, tok.bos)
	for c in doc
		haskey(tok.char2id, c) ||
			throw(ArgumentError("character '$c' is not in the vocabulary"))
		push!(token_ids, tok.char2id[c])
	end
	push!(token_ids, tok.bos)
	return token_ids
end

"""
	decode(tok::Tokenizer, ids::AbstractVector{<:Integer}) -> String

Decode token IDs back into a string, stripping boundary tokens.
Throws `ArgumentError` if `ids` contains an ID outside the vocabulary.
"""
function decode(tok::Tokenizer, ids::AbstractVector{<:Integer})
	chars = Vector{Char}()
	for id in ids
		if id == tok.bos
			continue
		end
		checkbounds(Bool, tok.uchars, id) ||
			throw(ArgumentError("token ID $id is outside the vocabulary (1:$(tok.vocab_size))"))
		push!(chars, tok.uchars[id])
	end
	return join(chars)
end
