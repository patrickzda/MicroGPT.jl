# MicroGPT

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://patrickzda.github.io/MicroGPT.jl/dev/)
[![CI](https://github.com/patrickzda/MicroGPT.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/patrickzda/MicroGPT.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/patrickzda/MicroGPT.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/patrickzda/MicroGPT.jl)

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

### Usage

```julia
using MicroGPT

# Load the names dataset (downloads automatically on first run)
docs = load_data()

# Build a character-level tokenizer from the dataset
tok = Tokenizer(docs)

# Encode a name to token IDs (wrapped with boundary tokens), decode back
ids = encode(tok, "anna")
decode(tok, ids)

# Autograd: wrap scalars in Value, compute gradients
a = Value(2.0)
b = Value(3.0)
c = a * b + relu(a - b)
backward!(c)
a.grad, b.grad
```

### Running the tests

When MicroGPT.jl is installed as a package, run its test suite through the package
manager. There is no local `test/` directory to point at:

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
  tokenizer.jl   # character-level tokenizer (Tokenizer, encode, decode)
test/            # test suite and fixtures, run via test/runtests.jl
docs/            # documentation sources
```
