import Base: +, -, *, /
using LinearAlgebra

mutable struct AValue{T<:AbstractArray}
    data::T
    grad::T
    children::C
    local_grads::G
end

function AValue(data::T) where T<:AbstractArray
    AValue(data, zero(data), (), ())
end


#
# forward: y = W*x
# dx = W' * dy
# dW = dy * x'
#
function linear(x::AValue, W::AValue)
    y = W.data * x.data

    AValue(y,
           zero(y),
           (x, W),
           (dy -> W.data' * dy, dy -> dy * x.data'))
end

#
# Forward: p = exp(x - max(x)) / SUM exp(x - max(x)) = e / SUM(e)
# dx = p .* (dy - dot(dy, p))
#
function softmax(x::AValue)
    e = exp .(x.data .- maximum(x.data))
    p = e ./ sum(e)

    AValue(p,
           zero(p),
           (x,),
           (dy -> p .* (dy .- dot(dy, p)),))
end

#
# Forward: y = x * scale, scale = (mean(x^2) + 0.00..)^(-0.5)
# dx = scale * dy - x * (scale^3 / n) * dot(dy, x) --> dx calculated with LLM, Link to prompt: https://chatgpt.com/share/6a3d41a0-445c-83eb-ba53-876e591115d8
#
function rmsnorm(x::AValue; eps::Float64 = 1e-5): #e=0.000 
    ms = sum(xi * xi for xi in x.data) / len(x.data)
    scale = (ms + 1e-5) ** -0.5
    y = [xi * scale for xi in x.data]
    n = len(x.data)

    # x.data = scaled
    # grad = zero(y) --> x über
    AValue(y, 
    zero(y),
    (x,),
    (dy -> scale .* dy - x.data * (scale^3 / n) * dot(dy, x.data),))

    # (ms)