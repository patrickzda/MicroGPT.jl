# This file has been designed with the help of LLMs (Claude)
# Version-aware autograd benchmark.
# Usage:
#   julia --project=@bench profiling/ad_benchmark.jl [src/autograd.jl] [label]
#
# Forward and backward are benchmarked separately

using BenchmarkTools
using Random

const SRC   = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "src", "autograd.jl")
const LABEL = length(ARGS) >= 2 ? ARGS[2] : "current"

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

# Build the forward graph. On the tape version this records into a tape.
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

# A fresh loss node
fresh_loss() = HAS_TAPE ? build_recorded()[1] : build()

# Forward
fwd = HAS_TAPE ? (@benchmark build_recorded()) : (@benchmark build())

# backword (without tape)
bwd_rec = @benchmark AD.backward!(loss) setup = (zero_grads!(); loss = fresh_loss())

# backward (with tape)
bwd_tape = if HAS_TAPE
    @benchmark AD.backward!(loss, tape) setup = (zero_grads!(); (loss, tape) = build_recorded())
else
    nothing
end

# context
NODES = node_count(fresh_loss())

using Printf

ns(t)  = BenchmarkTools.minimum(t).time          # ns, min = least-noisy estimate
al(t)  = BenchmarkTools.minimum(t).allocs
mem(t) = BenchmarkTools.minimum(t).memory        # bytes

fmt_mem(b) = b < 1024   ? @sprintf("%d B", b)        :
             b < 1024^2 ? @sprintf("%.1f KiB", b/1024) :
                          @sprintf("%.2f MiB", b/1024^2)

# One row per pass: min/median/mean time (µs), allocations, peak memory.
# `bs` is one or more benchmarks whose stats are summed (so a forward+backward
# total combines the two passes).
function stat_row(name, bs...)
    tmin = sum(BenchmarkTools.minimum(b).time for b in bs)
    tmed = sum(BenchmarkTools.median(b).time  for b in bs)
    tmean = sum(BenchmarkTools.mean(b).time   for b in bs)
    (
        name,
        @sprintf("%.2f", tmin  / 1e3),
        @sprintf("%.2f", tmed  / 1e3),
        @sprintf("%.2f", tmean / 1e3),
        string(sum(al(b)  for b in bs)),
        fmt_mem(sum(mem(b) for b in bs)),
    )
end

function print_table(header, rows)
    cols   = length(header)
    widths = [maximum(length(r[i]) for r in vcat(Any[header], rows)) for i in 1:cols]
    line(r) = join((rpad(r[i], widths[i]) for i in 1:cols), "  ")
    println(line(header))
    println(join(("-"^widths[i] for i in 1:cols), "  "))
    foreach(r -> println(line(r)), rows)
end

rows = Any[stat_row("forward", fwd), stat_row("backward (recursive)", bwd_rec)]
HAS_TAPE && push!(rows, stat_row("backward (tape)", bwd_tape))
push!(rows, stat_row("total (fwd + bwd recursive)", fwd, bwd_rec))
HAS_TAPE && push!(rows, stat_row("total (fwd + bwd tape)", fwd, bwd_tape))

println()
println("Autograd benchmark — label=$LABEL  nodes=$NODES  ",
        "(dim=$DIM, layers=$LAYERS, tape=$HAS_TAPE)")
print_table(("pass", "min (µs)", "median (µs)", "mean (µs)", "allocs", "memory"), rows)
println()

