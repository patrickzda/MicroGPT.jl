# This file has been designed with the help of LLMs (Claude)
# Version-aware autograd profiler
#
# Usage:
#   julia --project=@prof profiling/ad_profile.jl [src/autograd.jl] [label]
#
# The @prof environment needs Profile (stdlib) and ProfileSVG:
#   julia --project=@prof -e 'using Pkg; Pkg.add("ProfileSVG")'

using Profile
using ProfileSVG
using Random

const SRC   = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "src", "autograd.jl")
const LABEL = length(ARGS) >= 2 ? ARGS[2] : "current"
const OUTDIR = @__DIR__

module AD
end
Base.include(AD, abspath(SRC))

const HAS_TAPE = isdefined(AD, :record!)

# Workload: a deep stack of (W*x + b -> relu) so node count is high enough that
# graph-traversal cost in the backward pass is actually measurable.
const DIM    = 128
const LAYERS = 32

Random.seed!(42)
const Ws = [AD.AValue(randn(DIM, DIM) ./ sqrt(DIM)) for _ in 1:LAYERS]
const bs = [AD.AValue(randn(DIM))                   for _ in 1:LAYERS]
const x0 = AD.AValue(randn(DIM))

zero_grads!() = for p in Iterators.flatten((Ws, bs, (x0,)))
    fill!(p.grad, 0)
end

function build()
    h = x0
    for i in 1:LAYERS
        h = AD.relu(Ws[i] * h + bs[i])
    end
    return AD.sum(h)
end

function build_recorded()
    local loss
    tape = AD.record!(() -> (loss = build()))
    return loss, tape
end

function node_count(root)
    seen = Set{UInt}()
    stack = Any[root]
    while !isempty(stack)
        n = pop!(stack)
        id = objectid(n.grad)
        id in seen && continue
        push!(seen, id)
        for p in n.parents
            push!(stack, p)
        end
    end
    return length(seen)
end

# One full forward+backward step (what we profile).
function step!()
    zero_grads!()
    if HAS_TAPE
        loss, tape = build_recorded()
        AD.backward!(loss, tape)
    else
        loss = build()
        AD.backward!(loss)
    end
    return nothing
end

# Warm up
print("warming up (compiling)... "); flush(stdout)
step!()
println("done"); flush(stdout)

const NODES = node_count(HAS_TAPE ? build_recorded()[1] : build())

stats  = @timed step!()
allocs = @allocations step!()

const ITERS = 2000
print("profiling $(ITERS) steps... "); flush(stdout)
Profile.clear()
Profile.@profile for _ in 1:ITERS
    step!()
end
println("done"); flush(stdout)

const FLAME = joinpath(OUTDIR, "ad_profile_$(LABEL).svg")
print("saving flame graph... "); flush(stdout)
ProfileSVG.save(FLAME; maxdepth = 200, maxframes = 10000, width = 2400)
println("done"); flush(stdout)

# --- report ---
println("PROFILE\t", LABEL)
println("  nodes        : ", NODES)
println("  time (1 step): ", round(stats.time * 1e6, digits = 2), " µs")
println("  gc time      : ", round(stats.gctime * 1e6, digits = 2), " µs")
println("  allocations  : ", allocs)
println("  memory       : ", stats.bytes, " bytes")
println("  flame graph  : ", FLAME, "  (", ITERS, " steps sampled)")
