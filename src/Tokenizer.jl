module Tokenizer

export tokenizer_JuML

function tokenizer_JuML(docs::Vector{String})
	uchars = sort(collect(Set(join(docs))))
	BOS = length(uchars)
	vocab_size = length(uchars) + 1
#	@info "vocab size: $vocab_size"

	function encode(doc::String)
		token_ids = Vector{Int}()
		push!(token_ids, BOS)
		for c in doc
			id = findfirst(==(c), uchars) - 1  # Julia 1  > Token ID 0
			push!(token_ids, id)
		end
		push!(token_ids, BOS)
		return token_ids
	end

	function decode(ids::Vector{Int})
		chars = Vector{Char}()
		for id in ids
			if id != BOS
				push!(chars, uchars[id + 1])  # Token ID 0 > Julia 1
			end
		end
		return join(chars)
	end

	return uchars, BOS, vocab_size, encode, decode
end

end