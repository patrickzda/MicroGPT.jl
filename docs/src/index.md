```@meta
CurrentModule = MicroGPT
```

# MicroGPT

Documentation for [MicroGPT](https://github.com/patrickzda/MicroGPT.jl).

MicroGPT.jl is a minimal, educational re-implementation of the building blocks of a GPT-style model in pure Julia, inspired by Andrej Karpathys [microgpt](https://karpathy.github.io/2026/02/12/microgpt/):
a scalar reverse-mode autograd engine, a
character-level tokenizer, and a small dataset loader.

## Getting started

Install the package directly from GitHub (press `]` in the REPL to enter package
mode):

```julia
pkg> add https://github.com/patrickzda/MicroGPT.jl
```

From here, see the [Tokenizer](@ref) and [Autograd](@ref) guides for usage
examples, and the [API reference](@ref) for the full list of exported functions.
