```@meta
CurrentModule = MicroGPT
```

# GPT

A small GPT-style language model built on the [Autograd](autograd.md) engine,
the [Adam](optimizer.md) optimizer and the [Tokenizer](tokenizer.md). Each
parameter matrix is a single [`AValue`](@ref) node.

## Configuration

A model is described by a [`GPTConfig`](@ref): the vocabulary size, embedding
width, number of attention heads, number of transformer layers and the maximum
sequence length (`block_size`). [`head_dim`](@ref) gives the per-head width,
`n_embd ÷ n_head`.

```@example gpt
using MicroGPT

docs = ["emma", "olivia", "ava", "isabella", "sophia"]
tok = Tokenizer(docs)

config = GPTConfig(;
    vocab_size=length(tok.uchars) + 1,
    n_embd=16,
    n_head=2,
    n_layer=2,
    block_size=16,
)
head_dim(config)
```

## Building a model

[`GPT`](@ref) builds a model with randomly initialised weights bound to a
[`Tokenizer`](@ref). It holds the config, a `state_dict` mapping weight names to
matrices, a flat `params` list for the optimizer, and the tokenizer:

```@example gpt
model = GPT(config, tok)
length(model.params)
```

## Training

[`train!`](@ref) trains the model in place on a list of documents with a single
[`Adam`](@ref) optimizer over all weight matrices. Each step forwards one
document, accumulates the per-position cross-entropy loss, backpropagates with
[`backward!`](@ref) and takes one [`step!`](@ref):

```julia
train!(model, docs; num_steps=1000, learning_rate=0.01)
```

The `use_tape` keyword selects how [`backward!`](@ref) finds its topological order: with the
tape (the default) the backward pass is a flat reverse walk over the recorded
operations instead of a recursive sort of the graph.

## Generating

[`generate`](@ref) samples one sequence from the model and decodes it back to a
string, stopping at the BOS token or `block_size`. `temperature` controls how
sharp the sampling distribution is:

```julia
generate(model; temperature=0.5)
```

## Saving and loading

[`save_model`](@ref) saves the config, the raw weight matrices and the
tokenizer vocabulary to a file. [`load_model`](@ref) reconstructs an equivalent
[`GPT`](@ref), rebuilding the tokenizer and wrapping each weight matrix back into
an [`AValue`](@ref):

```julia
save_model("model.jls", model)
model = load_model("model.jls")
```
