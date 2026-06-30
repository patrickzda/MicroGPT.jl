module MicroGPT

using Downloads: download
using Random: shuffle!, default_rng
using LinearAlgebra: dot
using Serialization: serialize, deserialize

include("autograd.jl")
include("dataloader.jl")
include("tokenizer.jl")
include("optimizer.jl")
include("gpt.jl")

export load_data, Tokenizer, encode, decode
export AValue, backward!, relu
export Adam, step!, zero_grad!
export mul_elementwise, div_elementwise, pow_elementwise_scalar
export linear, softmax, rmsnorm
export GPTConfig, GPT, head_dim, train!, generate, save_model, load_model

end
