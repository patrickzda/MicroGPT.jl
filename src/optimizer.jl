# Interface for the optimizer
# zero_grad! and step!
abstract type Optimizer end

"""
    zero_grad!(opt::Optimizer)

Set the shared gradients of `opt` to zero.
"""
function zero_grad!(opt::Optimizer)
    for p in opt.params
        _zero_grad!(p)
    end
end
_zero_grad!(p::AValue) = fill!(p.grad, zero(eltype(p.grad))) # Cannot assignt to immutable struct


"""
    Adam <: Optimizer

State for an Adam optimizer over a flat list of `AValue` parameters.

Holds the parameters being optimized along with the hyperparameters and the
per-parameter moment buffers that `Àdam` maintains across `step!` calls.
"""
mutable struct Adam{T<:AValue,A<:AbstractArray,F<:AbstractFloat} <: Optimizer
    params::Vector{T}           # parameters to optimize
    α::F                        # learning rate
    β1::F                       # first-moment decay
    β2::F                       # second-moment decay
    ϵ::F                        # numerical stabilizer
    m::Vector{A}                # 1st moment estimate
    v::Vector{A}                # 2nd moment buffer
    t::Int                      # timestep, for bias correction
end

"""
    Adam(params; α=0.01, β1=0.85, β2=0.99, ϵ=1e-8)

Build an Adam optimizer over a flat list of `AValue` parameters.
"""
function Adam(params::Vector{<:AValue}; α=0.01, β1=0.85, β2=0.99, ϵ=1e-8)
    m = [zero(p.data) for p in params]
    v = [zero(p.data) for p in params]
    α, β1, β2, ϵ = promote(α, β1, β2, ϵ)   # one concrete F for all four
    return Adam(params, α, β1, β2, ϵ, m, v, 0)
end

"""
    step!(opt::Adam)
One `Adam` update step over all `AValue` parameters, elementwise per array.
Assumes gradients have been updated before executing.
"""
function step!(opt::Adam)
    opt.t += 1
    for (i, p) in enumerate(opt.params)
        @. opt.m[i] = opt.β1 * opt.m[i] + (1 - opt.β1) * p.grad
        @. opt.v[i] = opt.β2 * opt.v[i] + (1 - opt.β2) * p.grad^2
        m̂ = opt.m[i] ./ (1 - opt.β1^opt.t)
        v̂ = opt.v[i] ./ (1 - opt.β2^opt.t)
        @. p.data -= opt.α * m̂ / (sqrt(v̂) + opt.ϵ)
    end
end