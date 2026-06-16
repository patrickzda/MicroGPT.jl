module MicroGPT


using Downloads
using Random

include("autograd.jl")
include("dataloader.jl")
include("tokenizer.jl")




export load_data, Tokenizer, encode, decode
export Value, backward!, relu

end
