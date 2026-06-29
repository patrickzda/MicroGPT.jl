# MicroGPT

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://patrickzda.github.io/MicroGPT.jl/dev/)
[![CI](https://github.com/patrickzda/MicroGPT.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/patrickzda/MicroGPT.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/patrickzda/MicroGPT.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/patrickzda/MicroGPT.jl)

**Requirements:** Julia 1.11

## Getting Started (as a User)

Use this guide if you just want to install MicroGPT.jl and call it from your own
code.

### Installation

MicroGPT.jl is not yet registered. Install it directly from GitHub using Julia's
package manager (press `]` in the Julia REPL to enter package mode):

```julia
pkg> add https://github.com/patrickzda/MicroGPT.jl
```

### Training a GPT

The snippet below trains a small character-level GPT on the names dataset,
generates samples, and saves/loads the trained model. It is the same runnable
[`run.jl`](run.jl) script at the project root, which you can execute with:

```bash
julia --project=. run.jl
```

```julia
using MicroGPT

# Load the dataset (downloads to `input.txt` on first run) and build a tokenizer
docs = load_data("input.txt")
tokenizer = Tokenizer(docs)
println("num docs: $(length(docs)) | vocab size: $(tokenizer.vocab_size)")

# Configure and create the model
config = GPTConfig(
    vocab_size = tokenizer.vocab_size,
    n_embd     = 16,
    n_head     = 4,
    n_layer    = 1,
    block_size = 16,
)
model = GPT(config, tokenizer)

# Train
train!(model, docs; num_steps = 2000, learning_rate = 0.01)

# Generate some samples
println("\nSamples:")
for _ in 1:20
    println("  ", generate(model; temperature = 0.8))
end

# Save and reload the trained model
save_model("model.jls", model)
new_model = load_model("model.jls")
```

### Running the tests

When MicroGPT.jl is installed as a package, run its test suite through the package manager:

```julia
pkg> test MicroGPT
```

Or, equivalently, from a script or the REPL:

```julia
using Pkg
Pkg.test("MicroGPT")
```

## Getting Started (as a Developer)

Use this guide if you want to work on MicroGPT.jl itself, read the source, run
the tests against a checkout, or contribute changes.

### Clone the repository

```bash
git clone https://github.com/patrickzda/MicroGPT.jl
cd MicroGPT.jl
```

### Set up the environment

Instantiate the package's dependencies from the checkout:

```julia
pkg> activate .
pkg> instantiate
```

### Running the tests

From the repository root, run the test suite directly against the checkout. The
tests have their own environment under `test/`, so activate it when invoking the
runner:

```bash
julia --project=test test/runtests.jl
```
or:

```julia
(MicroGPT) pkg> test
```


### Project layout

```
src/
  MicroGPT.jl    # module entry point, exports the public API
  autograd.jl    # scalar reverse-mode autograd (Value, backward!, relu)
  dataloader.jl  # dataset loading (load_data)
  gpt.jl         # gpt model (layers, train, inference)
  optimizer.jl   # Adam optimizer
  tokenizer.jl   # character-level tokenizer (Tokenizer, encode, decode)
test/            # test suite and fixtures, run via test/runtests.jl
docs/            # documentation sources
```

## AI / LLM usage

Large language models (e.g. ChatGPT / GitHub Copilot / Claude) were used as
assistants during development of this project, for example to draft and refine
documentation, tests, and parts of the source code. All AI-assisted output was
reviewed and edited by the authors.

## Data

The dataset of names used by `load_data` comes from Andrej Karpathy's
[makemore](https://github.com/karpathy/makemore) project and is redistributed
under the MIT License. A copy bundled with the test suite lives at
`test/names.txt`, with the accompanying license at `test/names.LICENSE`.
