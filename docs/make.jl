using MicroGPT
using Documenter

DocMeta.setdocmeta!(MicroGPT, :DocTestSetup, :(using MicroGPT); recursive=true)

makedocs(;
    modules=[MicroGPT],
    authors="Patrick Zdanowski p.zdanowski@campus.tu-berlin.de",
    sitename="MicroGPT.jl",
    format=Documenter.HTML(;
        canonical="https://patrickzda.github.io/MicroGPT.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Tokenizer" => "tokenizer.md",
        "Autograd" => "autograd.md",
        "Optimizer" => "optimizer.md",
        "GPT" => "gpt.md",
        "API reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/patrickzda/MicroGPT.jl",
    devbranch="main",
)
