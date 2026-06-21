using MicroGPT
using Documenter

DocMeta.setdocmeta!(MicroGPT, :DocTestSetup, :(using MicroGPT); recursive=true)

makedocs(;
    modules=[MicroGPT],
    authors="Patrick Zdanowski p.zdanowski@campus.tu-berlin.de",
    sitename="MicroGPT.jl",
    format=Documenter.HTML(;
        canonical="https://patrickzda.github.io/MicroGPT.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/patrickzda/MicroGPT.jl",
    devbranch="main",
)
