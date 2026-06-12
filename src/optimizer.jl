# Interface for the optimizer
# zero_grad! and step!
abstract type Optimizer end

"""
    zero_grad!(opt::Optimizer)

Set the shared gradients of `opt` to zero.
"""
function zero_grad!(opt::Optimizer)
    for p in opt.params
        p.grad = zero(p.data)
    end
end

# Parameters for the Adam Optimizer
mutable struct Adam <: Optimizer
    params::Vector{<:Value}       # parameters to optimize
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
function Adam(params::Vector{<:Value}; α=0.01, β1=0.85, β2=0.99, ϵ=1e-8)   # TODO Idk if i should use the greek letters or this, for simplicity of config maybe???
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

