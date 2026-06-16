const NAMES_URL = "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"

function load_data(path::String="input.txt"; shuffle::Bool=true, rng=GLOBAL_RNG)::Vector{String}
    if !isfile(path)
        #		@info "Downloading names dataset to $path ..."
        download(NAMES_URL, path)
    end
    docs = [strip(l) for l in eachline(path) if !isempty(strip(l))]
    #	Filter auf a-z?
    #	docs = filter(doc -> all(c -> 'a' <= c <= 'z', doc), docs)  # nur a-z
    shuffle && shuffle!(rng, docs)
    #	@info "num docs: $(length(docs))"
    return docs
end

