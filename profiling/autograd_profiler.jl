using Profile
using ProfileView
using Random
using MicroGPT

const DIM_IN = 256
const DIM_OUT = 256
const STEPS = 1000

Random.seed!(42)

function zero_grad!(W, b, x)
    fill!(W.grad, zero(eltype(W.grad)))
    fill!(b.grad, zero(eltype(b.grad)))
    fill!(x.grad, zero(eltype(x.grad)))
end

function forward(W, b, x)
    output = W * x + b
    return relu(output)
end

function compute_one_step!(W, b, x)
    y = forward(W, b, x)

    loss = sum(y)
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

        for parent in node.parents
            visit(parent)
        end
    end

    visit(root)
    return length(visited_set)
end

W = AValue(randn(DIM_OUT, DIM_IN))
b = AValue(randn(DIM_OUT))
x = AValue(randn(DIM_IN))

# Run once for compilation and count total nodes
zero_grad!(W, b, x)
loss = compute_one_step!(W, b, x)
println("Total nodes: ", count_nodes(loss))

Profile.clear()

# Generate profile view
ProfileView.@profview begin
    for i in 1:STEPS
        zero_grad!(W, b, x)
        compute_one_step!(W, b, x)
    end
end

# Measure execution time and allocations
zero_grad!(W, b, x)
@time begin
    compute_one_step!(W, b, x)
    nothing
end
