const NAMES_URL = "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"

"""
    load_data(path="input.txt"; shuffle=true, rng=Random.default_rng()) -> Vector{String}

Load a corpus of documents, one per non-empty line of `path`, with surrounding
whitespace stripped.

If `path` does not exist, the names dataset is downloaded to `path` first. When
`shuffle` is `true` the documents are shuffled in place using `rng`.
"""
function load_data(path::String="input.txt"; shuffle::Bool=true, rng=default_rng())
    if !isfile(path)
        @debug "Downloading names dataset to $path ..."
        download(NAMES_URL, path)
    end
    docs = [String(strip(l)) for l in eachline(path) if !isempty(strip(l))]
    shuffle && shuffle!(rng, docs)
    @debug "num docs: $(length(docs))"
    return docs
end

