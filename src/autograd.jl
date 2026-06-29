import Base: +, -, *, /, log, exp, sum, getindex, vcat, hcat, transpose
using LinearAlgebra: dot, mul!

"""
    AValue{T<:AbstractFloat, N}

Node in a vector-based automatic differentiation graph.

Each node stores its forward value `data` and accumulated 
gradient `grad`, its parent nodes `parents` and a pullback
function `pullback_fn` used during reverse-mode automatic differentiation.
"""
# Tape to speed up the backward / no recursive walk trough the graph.
const _TAPE = Base.RefValue{Union{Nothing,Vector}}(nothing)

struct AValue{T<:AbstractFloat,N}
    data::Array{T,N}
    grad::Array{T,N}
    parents::Tuple
    pullback_fn::Function

    # Construct a graph node with recording on tape
    function AValue{T,N}(data::Array{T,N}, grad::Array{T,N}, parents::Tuple, pullback_fn::Function) where {T<:AbstractFloat,N}
        node = new{T,N}(data, grad, parents, pullback_fn)
        tape = _TAPE[]
        (tape === nothing || isempty(parents)) || push!(tape, node)
        return node
    end
end

# Outer constructor that infers the type parameters
AValue(data::Array{T,N}, grad::Array{T,N}, parents::Tuple, pullback_fn::Function) where {T<:AbstractFloat,N} =
    AValue{T,N}(data, grad, parents, pullback_fn)

"""
    record!(f) -> tape

Run forward pass `f` with tape recording enabled and return the tape (the nodes
created, in topological order).
"""
function record!(f)
    prev = _TAPE[]
    tape = AValue[]
    _TAPE[] = tape
    try
        f()
    finally
        _TAPE[] = prev
    end
    return tape
end


"""
    AValue(val::AbstractArray)

Constructor for an AValue leaf node.

Returns a node holding value `val`, zero grads, no parents and an empty
pullback function. Integer arrays are promoted to floats so gradients are
always float-backed.
"""
function AValue(data::Array{T,N}) where {T<:AbstractFloat,N}
    return AValue(data, zero(data), (), _ -> ())
end
AValue(data::AbstractArray{<:Real}) = AValue(float.(collect(data)))

# Broadcasting over a 0-dimensional array can collapse to a scalar; re-wrap it.
function AValue(data::Real, grad::Real, parents::Tuple, pullback_fn::Function)
    return AValue(fill(float(data)), fill(float(grad)), parents, pullback_fn)
end

function +(a::AValue, b::AValue)
    if size(a.data) != size(b.data)
        throw(DimensionMismatch("Both arguments need the same shape."))
    end

    output = a.data + b.data

    return AValue(
        output,
        zero(output),
        (a, b),
        dc -> begin
            a.grad .+= dc
            b.grad .+= dc
        end
    )
end

function +(a::AValue, b::Real)
    output = a.data .+ b

    return AValue(
        output,
        zero(output),
        (a,),
        dc -> (a.grad .+= dc)
    )
end

function +(a::Real, b::AValue)
    return b + a
end

function -(a::AValue, b::AValue)
    if size(a.data) != size(b.data)
        throw(DimensionMismatch("Both arguments must have the same shape."))
    end

    output = a.data - b.data

    return AValue(
        output,
        zero(output),
        (a, b),
        dc -> begin
            a.grad .+= dc
            b.grad .-= dc
        end
    )
end

function -(a::AValue, b::Real)
    output = a.data .- b

    return AValue(
        output,
        zero(output),
        (a,),
        dc -> (a.grad .+= dc)
    )
end

function -(a::Real, b::AValue)
    return -b + a
end

function -(a::AValue)
    output = a.data .* -1

    return AValue(
        output,
        zero(output),
        (a,),
        dc -> (a.grad .-= dc)
    )
end

# Matrix multiply: matrix × vector and matrix × matrix in one method.
# d(A*B): A.grad += dc * B', B.grad += A' * dc, accumulated in place with `mul!`.
function *(a::AValue{<:AbstractFloat,2}, b::Union{AValue{<:AbstractFloat,1},AValue{<:AbstractFloat,2}})
    output = a.data * b.data

    return AValue(
        output,
        zero(output),
        (a, b),
        dc -> begin
            mul!(a.grad, dc, transpose(b.data), true, true)   # a.grad += dc * b'
            mul!(b.grad, transpose(a.data), dc, true, true)   # b.grad += a' * dc
        end
    )
end

function *(a::AValue{T}, b::Real) where {T}
    c = T(b)
    output = a.data .* c

    return AValue(
        output,
        zero(output),
        (a,),
        dc -> (a.grad .+= c .* dc)
    )
end

function *(a::Real, b::AValue)
    return b * a
end

"""
    mul_elementwise(a::AValue, b::AValue)

Multiplies two AValues (`a` and `b`) elementwise. Both arguments
must be of same shape.

Returns a new AValue type holding the elementwise multiplication.
"""
function mul_elementwise(a::AValue, b::AValue)
    if size(a.data) != size(b.data)
        throw(DimensionMismatch("Both arguments must have the same shape."))
    end

    output = a.data .* b.data
    return AValue(
        output,
        zero(output),
        (a, b),
        dc -> begin
            a.grad .+= dc .* b.data
            b.grad .+= dc .* a.data
        end
    )
end

function /(a::AValue, b::Real)
    output = a.data ./ b

    return AValue(
        output,
        zero(output),
        (a,),
        dc -> (a.grad .+= dc ./ b)
    )
end

"""
    div_elementwise(a::AValue, b::AValue)

Divides two AValues (`a` and `b`) elementwise. Both arguments
must be of same shape.

Returns a new AValue type holding the elementwise division.
"""
function div_elementwise(a::AValue, b::AValue)
    if size(a.data) != size(b.data)
        throw(DimensionMismatch("Both arguments must have the same shape."))
    end

    output = a.data ./ b.data
    return AValue(
        output,
        zero(output),
        (a, b),
        dc -> begin
            a.grad .+= dc ./ b.data
            b.grad .+= -dc .* a.data ./ (b.data .* b.data)
        end
    )
end

"""
    pow_elementwise_scalar(a::AValue, b::Real)

Potentiates an AValue `a` by a scalar `b` elementwise.

Returns a new AValue type holding the elementwise potentiation.
"""
function pow_elementwise_scalar(a::AValue, b::Real)
    output = a.data .^ b
    return AValue(
        output,
        zero(output),
        (a,),
        dc -> (a.grad .+= dc .* b .* (a.data .^ (b - 1)))
    )
end

function log(a::AValue)
    output = log.(a.data)

    return AValue(
        output,
        zero(output),
        (a,),
        dc -> (a.grad .+= dc ./ a.data)
    )
end

function exp(a::AValue)
    output = exp.(a.data)

    return AValue(
        output,
        zero(output),
        (a,),
        dc -> (a.grad .+= dc .* output)
    )
end

function sum(a::AValue{<:AbstractFloat,1})
    output = fill(sum(a.data))

    return AValue(
        output,
        zero(output),
        (a,),
        dc -> (a.grad .+= dc[])
    )
end

"""
    relu(a::AValue)

Applies elementwise ReLU to AValue `a`.

Returns a new AValue type holding the elementwise ReLU.
"""
function relu(a::AValue)
    output = max.(zero(eltype(a.data)), a.data)

    return AValue(
        output,
        zero(output),
        (a,),
        dc -> (a.grad .+= dc .* (a.data .> 0))
    )
end

#
# forward: y = W*x
# dx = W' * dy
# dW = dy * x'
#
"""
    linear(x::AValue, W::AValue)

Applies a linear transformation to AValue `x` using weight matrix `W`.

Returns a new AValue type holding the transformed output. (Hint: Docstring generated with https://chatgpt.com/share/6a3fcb2d-0ecc-83eb-9e40-c00c1661c295)
"""
function linear(x::AValue, W::AValue)
    y = W.data * x.data

    AValue(
        y,
        zero(y),
        (x, W),
        dy -> begin
            mul!(x.grad, transpose(W.data), dy, true, true)   # x.grad += W' * dy
            mul!(W.grad, dy, transpose(x.data), true, true)   # W.grad += dy * x'
        end
    )
end

#
# Forward: p = exp(x - max(x)) / SUM exp(x - max(x)) = e / SUM(e)
# dx = p .* (dy - dot(dy, p))
#
"""
    softmax(x::AValue{<:AbstractFloat, 1})

Applies the softmax function to AValue vector `x`.

Returns a new AValue type holding the softmax probabilities. (Hint: Docstring generated with https://chatgpt.com/share/6a3fcb2d-0ecc-83eb-9e40-c00c1661c295)
"""
function softmax(x::AValue{<:AbstractFloat,1})
    e = exp.(x.data .- maximum(x.data))
    p = e ./ sum(e)

    AValue(
        p,
        zero(p),
        (x,),
        dy -> (x.grad .+= p .* (dy .- dot(dy, p)))
    )
end

#
# Forward: y = x * scale, scale = (mean(x^2) + 0.00..)^(-0.5)
# dx = scale * dy - x * (scale^3 / n) * dot(dy, x) --> dx calculated with LLM, Link to prompt: https://chatgpt.com/share/6a3d41a0-445c-83eb-ba53-876e591115d8
#e=0.000
"""
    rmsnorm(x::AValue{<:AbstractFloat, 1}; eps::Float64 = 1e-5)

Applies RMS normalization to AValue vector `x`.

Returns a new AValue type holding the RMS-normalized vector. (Hint: Docstring generated with https://chatgpt.com/share/6a3fcb2d-0ecc-83eb-9e40-c00c1661c295)
"""
function rmsnorm(x::AValue{<:AbstractFloat,1}; eps::Float64=1e-5)
    n = length(x.data)
    ms = sum(x.data .^ 2) / n
    scale = (ms + eps) ^ -0.5
    y = x.data .* scale

    AValue(
        y,
        zero(y),
        (x,),
        #  scale, scale^3/n, dot(dy, x.data) : skalar
        #  dy, x.data: vector
        # out: vector
        dy -> (x.grad .+= scale .* dy .- x.data .* (scale^3 / n) .* dot(dy, x.data))
    )
end

"""
    getindex(a::AValue, inds...)

Index into AValue `a` (e.g. `a[i, :]` to pick a matrix row, or `a[i:j]` to slice
a vector). Scalar results are wrapped as a 0-dimensional array so they stay
AValues. The gradient is scattered back into the selected positions of `a`.
"""
function getindex(a::AValue, inds...)
    raw = a.data[inds...]
    output = raw isa AbstractArray ? raw : fill(raw)

    return AValue(
        output,
        zero(output),
        (a,),
        dc -> (view(a.grad, inds...) .+= raw isa AbstractArray ? dc : dc[])
    )
end

"""
    vcat(ts::AValue...)

Concatenate vector AValues into one longer vector AValue (e.g. multi-head
attention outputs). Gradients are split back to each input in order.
"""
function vcat(ts::AValue...)
    output = vcat((t.data for t in ts)...)

    function pullback(dc)
        offset = 0
        for t in ts
            n = length(t.data)
            t.grad .+= @view dc[(offset+1):(offset+n)]
            offset += n
        end
    end

    return AValue(output, zero(output), ts, pullback)
end

"""
    hcat(ts::AValue...)

Stack vector AValues as the columns of one matrix AValue (e.g. the cached
attention keys/values of a head, giving an `hd × T` matrix). Gradients flow back
column-wise to each input.
"""
function hcat(ts::AValue...)
    output = stack(t.data for t in ts)

    function pullback(dc)
        for (j, t) in enumerate(ts)
            t.grad .+= @view dc[:, j]
        end
    end

    return AValue(output, zero(output), ts, pullback)
end

"""
    transpose(a::AValue{<:AbstractFloat, 2})

Transpose a matrix AValue. The output gradient is routed back transposed.
"""
function transpose(a::AValue{<:AbstractFloat,2})
    output = Array(transpose(a.data))
    return AValue(output, zero(output), (a,), dc -> (a.grad .+= transpose(dc)))
end

"""
    backward!(v::AValue, tape=nothing)

Computes the gradients of `v` with respect to every `AValue` in its computation
graph via backpropagation.

Pass the `tape` returned by [`record!`](@ref) (with `v` as its last node) to take
the flat reverse walk over it.
"""
function backward!(v::AValue, tape::Union{Nothing,Vector}=nothing)
    fill!(v.grad, one(eltype(v.grad)))

    if tape === nothing
        # No tape: build a topological order by recursively walking the graph
        topo = Any[]
        visited = IdSet{Any}()

        function build_topo(node)
            if !(node.grad in visited)
                push!(visited, node.grad)
                for parent in node.parents
                    build_topo(parent)
                end
                push!(topo, node)
            end
        end

        build_topo(v)
        for node in reverse(topo)
            node.pullback_fn(node.grad)
        end
    else
        # Tape already holds the nodes in topological order, just walk it backwards
        for i in lastindex(tape):-1:firstindex(tape)
            tape[i].pullback_fn(tape[i].grad)
        end
    end

    return nothing
end
