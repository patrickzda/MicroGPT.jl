using MicroGPT
using Test

@testset "MicroGPT.jl" begin
    include("tokenizer_tests.jl")
    include("dataloader_tests.jl")
    include("autograd_tests.jl")
end
