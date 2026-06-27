module MicroGPT

using Downloads: download
using Random: shuffle!, default_rng

include("autograd.jl")
include("dataloader.jl")
include("tokenizer.jl")
include("optimizer.jl")

export load_data, Tokenizer, encode, decode
export AValue, Value, backward!, relu
export Adam, step!, zero_grad!
export mul_elementwise, div_elementwise, pow_elementwise_scalar

end
