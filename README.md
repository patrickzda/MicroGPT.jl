# MicroGPT

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://patrickzda.github.io/MicroGPT.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://patrickzda.github.io/MicroGPT.jl/dev/)
[![Build Status](https://github.com/patrickzda/MicroGPT.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/patrickzda/MicroGPT.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/patrickzda/MicroGPT.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/patrickzda/MicroGPT.jl)

## Getting Started

**Requirements:** Julia 1.11

### Installation

MicroGPT.jl is not yet registered. Install directly from GitHub:

```julia
pkg> add https://github.com/patrickzda/MicroGPT.jl
```

### Usage

```julia
using MicroGPT

# Load the names dataset (downloads automatically on first run)
docs = dataloader_JuML()

# Build a character-level tokenizer from the dataset
uchars, BOS, vocab_size, encode, decode = tokenizer_JuML(docs)

# Encode a name to token IDs, decode back
ids = encode("anna")
decode(ids)

# Autograd: wrap scalars in Value, compute gradients
a = Value(2.0)
b = Value(3.0)
c = a * b + relu(a - b)
backward!(c)
a.grad, b.grad
```

### Running the tests

```
julia --project=test test/runtests.jl
```
