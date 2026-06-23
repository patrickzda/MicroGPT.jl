```@meta
CurrentModule = MicroGPT
```

# Tokenizer

Build a character-level [`Tokenizer`](@ref) from a corpus of documents, then
[`encode`](@ref) a string into token IDs and [`decode`](@ref) it back. The
following example runs during the documentation build:

```@example tokenizer
using MicroGPT

docs = ["emma", "olivia", "ava", "isabella", "sophia"]
tok = Tokenizer(docs)

ids = encode(tok, "ava")
```

```@example tokenizer
decode(tok, ids)
```

In your own code you can load the full names dataset with [`load_data`](@ref),
which downloads it automatically on first use:

```julia
docs = load_data()
tok = Tokenizer(docs)
```
