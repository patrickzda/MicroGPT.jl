using Profile
using ProfileView
using Random
using MicroGPT

const DIM_IN = 256
const DIM_OUT = 256
const STEPS = 100

Random.seed!(42)

function zero_grad!(W, b, x)
    for entry in W
        entry.grad = zero(entry.grad)
    end

    for entry in b
        entry.grad = zero(entry.grad)
    end

    for entry in x
        entry.grad = zero(entry.grad)
    end
end

function forward(W, b, x)
    output = [
        sum(W[i, j] * x[j] for j in 1:DIM_IN)
        for i in 1:DIM_OUT
    ]

    return relu.(output .+ b)
end

function compute_one_step!(W, b, x)
    y = forward(W, b, x)

    loss = sum(y; init=Value(0.0))
    backward!(loss)

    return loss
end

function count_nodes(root)
    visited_set = IdSet()

    function visit(node)
        if node in visited_set
            return
        end

        push!(visited_set, node)

        for child in node.children
            visit(child)
        end
    end

    visit(root)
    return length(visited_set)
end

W = [Value(randn()) for _ in 1:DIM_OUT, _ in 1:DIM_IN]
b = [Value(randn()) for _ in 1:DIM_OUT]
x = Value.(randn(DIM_IN))

# Run once for compilation and count total nodes
zero_grad!(W, b, x)
loss = compute_one_step!(W, b, x)
println("Total nodes: ", count_nodes(loss))

Profile.clear()

# Generate profile view
zero_grad!(W, b, x)
ProfileView.@profview compute_one_step!(W, b, x)

# Measure execution time and allocations
zero_grad!(W, b, x)
@time begin
    compute_one_step!(W, b, x)
    nothing
end
