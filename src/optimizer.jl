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
_zero_grad!(p::Value) = (p.grad = zero(p.data))
_zero_grad!(p::AValue) = fill!(p.grad, zero(eltype(p.grad))) # Cannot assignt to immutable struct


"""
    Adam <: Optimizer

State for an Adam optimizer over a flat list of `Value` parameters.

Holds the parameters being optimized along with the hyperparameters and the
per-parameter moment estimates that Adam maintains across `step!` calls.
"""
mutable struct Adam <: Optimizer
    params::Vector{<:Value}     # parameters to optimize
    α::AbstractFloat            # learning rate
    β1::AbstractFloat           # first-moment decay
    β2::AbstractFloat           # second-moment decay
    ϵ::AbstractFloat            # numerical stabilizer 
    m::Vector{AbstractFloat}    # 1st moment estimate 
    v::Vector{AbstractFloat}    # 2nd moment buffer
    t::Int                      # timestep,for bias correction
end

"""
    Adam(params; α=0.01, β1=0.85, β2=0.99, ϵ=1e-8)

Build an Adam optimizer over a flat list of `Value` parameters.
"""
function Adam(params::Vector{<:Value}; α=0.01, β1=0.85, β2=0.99, ϵ=1e-8)
    return Adam(params, α, β1, β2, ϵ,
        zeros(length(params)), zeros(length(params)), 0)
end

"""
    step!(opt::Adam)
One `Adam` update step over all `Value`'s. Updates each `Value`'s `data` field.
Assuemes gradients have been updated before executing.
"""
function step!(opt::Adam)
    opt.t += 1
    for (i, p) in enumerate(opt.params)
        opt.m[i] = opt.β1 * opt.m[i] + (1 - opt.β1) * p.grad
        opt.v[i] = opt.β2 * opt.v[i] + (1 - opt.β2) * p.grad^2
        m̂ = opt.m[i] / (1 - opt.β1^opt.t)
        v̂ = opt.v[i] / (1 - opt.β2^opt.t)
        p.data -= opt.α * m̂ / (v̂^0.5 + opt.ϵ)
    end
end


# Adam over `AValue` parameters (the vector-based engine in autograd.jl).
# Moment buffers are arrays matching each parameter's shape.
mutable struct AAdam{T<:AValue,A<:AbstractArray,F<:AbstractFloat} <: Optimizer
    params::Vector{T}
    α::F
    β1::F
    β2::F
    ϵ::F
    m::Vector{A}
    v::Vector{A}
    t::Int
end

"""
    AAdam(params; α=0.01, β1=0.85, β2=0.99, ϵ=1e-8)

Build an Adam optimizer over a flat list of `AValue` parameters. `step!` and
`zero_grad!` work exactly as for the scalar `Adam`.
"""
function AAdam(params::Vector{<:AValue}; α=0.01, β1=0.85, β2=0.99, ϵ=1e-8)
    m = [zero(p.data) for p in params]
    v = [zero(p.data) for p in params]
    α, β1, β2, ϵ = promote(α, β1, β2, ϵ)   # one concrete F for all four
    return AAdam(params, α, β1, β2, ϵ, m, v, 0)
end

"""
    step!(opt::AAdam)
One `Adam` update step over all `AValue` parameters, elementwise per array.
Assumes gradients have been updated before executing.
"""
function step!(opt::AAdam)
    opt.t += 1
    for (i, p) in enumerate(opt.params)
        @. opt.m[i] = opt.β1 * opt.m[i] + (1 - opt.β1) * p.grad
        @. opt.v[i] = opt.β2 * opt.v[i] + (1 - opt.β2) * p.grad^2
        m̂ = opt.m[i] ./ (1 - opt.β1^opt.t)
        v̂ = opt.v[i] ./ (1 - opt.β2^opt.t)
        @. p.data -= opt.α * m̂ / (sqrt(v̂) + opt.ϵ)
    end
end