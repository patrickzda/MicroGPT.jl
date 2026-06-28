import Base: +, -, *, /, log, exp, sum, getindex, vcat, hcat, transpose

"""
    AValue{D<:AbstractArray, G<:AbstractArray}

Node in a vector-based automatic differentiation graph.

Each node stores its forward value `data`, accumulated gradient `grad`,
its parent nodes `parents` and a pullback function `pullback_fn` that is
used during reverse-mode automatic differentiation.
"""
struct AValue{D<:AbstractArray, G<:AbstractArray}
    data::D
    grad::G
    parents::Tuple
    pullback_fn::Function
end


"""
    AValue{val::T}

Constructor for an AValue leaf node.

Returns a node holding value `val`, zero grads, no parents and empty 
pullback function.
"""
function AValue(val::T) where {T<:AbstractArray}
    return AValue(val, zero(val), (), _ -> ())
end

# Broadcasting over a 0-dimensional array collapses to a scalar
function AValue(data::Real, grad::Real, parents::Tuple, pullback_fn::Function)
    return AValue(fill(data), fill(grad), parents, pullback_fn)
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
        dc -> (dc, dc)
    )
end

function +(a::AValue, b::Real)
    output = a.data .+ b

    return AValue(
        output,
        zero(output),
        (a,),
        dc -> (dc,)
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
        dc -> (dc, -dc)
    )
end

function -(a::AValue, b::Real)
    output = a.data .- b

    return AValue(
        output,
        zero(output),
        (a,),
        dc -> (dc,)
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
        dc -> (-dc,)
    )
end

function *(a::AValue{<:AbstractMatrix}, b::AValue{<:AbstractMatrix})
    if size(a.data)[2] != size(b.data)[1]
        throw(DimensionMismatch("Number of columns must be equal to number of rows."))
    end

    output = a.data * b.data

    return AValue(
        output,
        zero(output),
        (a, b),
        dc -> (dc * transpose(b.data), transpose(a.data) * dc)
    )
end

function *(a::AValue{<:AbstractMatrix}, b::AValue{<:AbstractVector})
    if size(a.data)[2] != length(b.data)
        throw(DimensionMismatch("Number of matrix columns must be equal to vector size."))
    end

    output = a.data * b.data

    return AValue(
        output,
        zero(output),
        (a, b),
        dc -> (dc * transpose(b.data), transpose(a.data) * dc)
    )
end

function *(a::AValue, b::Real)
    output = a.data .* b

    return AValue(
        output,
        zero(output),
        (a,),
        dc -> (dc .* b,)
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
        dc -> (dc .* b.data, dc .* a.data)
    )
end

function /(a::AValue, b::Real)
    output = a.data ./ b

    return AValue(
        output,
        zero(output),
        (a,),
        dc -> (dc / b,)
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
        dc -> (dc ./ b.data, -dc .* a.data ./ (b.data .* b.data))
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
        dc -> (dc .* b .* (a.data .^ (b - 1)),)
    )
end

function log(a::AValue)
    output = log.(a.data)

    return AValue(
        output,
        zero(output),
        (a,),
        dc -> (dc ./ a.data,)
    )
end

function exp(a::AValue)
    output = exp.(a.data)

    return AValue(
        output,
        zero(output),
        (a,),
        dc -> (dc .* output,)
    )
end

function sum(a::AValue{<:AbstractVector})
    output = fill(sum(a.data))

    return AValue(
        output,
        zero(output),
        (a,),
        dc -> (fill(dc[], size(a.data)),)
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
        dc -> (dc .* (a.data .> 0),)
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
        dy -> (W.data' * dy, dy * x.data')
    )
end

#
# Forward: p = exp(x - max(x)) / SUM exp(x - max(x)) = e / SUM(e)
# dx = p .* (dy - dot(dy, p))
#
"""
    softmax(x::AValue{<:AbstractVector})

Applies the softmax function to AValue vector `x`.

Returns a new AValue type holding the softmax probabilities. (Hint: Docstring generated with https://chatgpt.com/share/6a3fcb2d-0ecc-83eb-9e40-c00c1661c295)
"""
function softmax(x::AValue{<:AbstractVector})
    e = exp.(x.data .- maximum(x.data))
    p = e ./ sum(e)

    AValue(
        p,
        zero(p),
        (x,),
        dy -> (p .* (dy .- dot(dy, p)),)
    )
end

#
# Forward: y = x * scale, scale = (mean(x^2) + 0.00..)^(-0.5)
# dx = scale * dy - x * (scale^3 / n) * dot(dy, x) --> dx calculated with LLM, Link to prompt: https://chatgpt.com/share/6a3d41a0-445c-83eb-ba53-876e591115d8
#e=0.000 
"""
    rmsnorm(x::AValue{<:AbstractVector}; eps::Float64 = 1e-5)

Applies RMS normalization to AValue vector `x`.

Returns a new AValue type holding the RMS-normalized vector. (Hint: Docstring generated with https://chatgpt.com/share/6a3fcb2d-0ecc-83eb-9e40-c00c1661c295)
"""
function rmsnorm(x::AValue{<:AbstractVector}; eps::Float64 = 1e-5)
    n = length(x.data)
    ms = sum(x.data .^ 2) / n
    scale = (ms + eps) ^ -0.5
    y = x.data .* scale

    # x.data = scaled
    # grad = zero(y) --> x über
    
    AValue(
        y, 
        zero(y),
        (x,),
        #(dy -> scale .* dy - x.data * (scale^3 / n) * dot(dy, x.data),))
        #  scale, scale^3/n, dot(dy, x.data) : skalar
        #  dy, x.data: vector
        # out: vector
        dy -> (scale .* dy .- x.data .* (scale^3 / n) .* dot(dy, x.data),)
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
        function (dc)
            g = zero(a.data)
            gview = @view g[inds...]
            gview .+= raw isa AbstractArray ? dc : dc[]
            (g,)
        end
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
        return map(ts) do t
            n = length(t.data)
            g = dc[offset+1:offset+n]
            offset += n
            g
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

    pullback(dc) = ntuple(j -> dc[:, j], length(ts))

    return AValue(output, zero(output), ts, pullback)
end

"""
    transpose(a::AValue{<:AbstractMatrix})

Transpose a matrix AValue. The output gradient is routed back transposed.
"""
function transpose(a::AValue{<:AbstractMatrix})
    output = Array(transpose(a.data))
    return AValue(output, zero(output), (a,), dc -> (Array(transpose(dc)),))
end

"""
    backward!(v::AValue)

Computes the gradients of `v` with respect to every `AValue` in its computation
graph via backpropagation.

This mutates all gradients in the computational graph. Gradients
are accumulated, so zeroing the gradients is required between passes.
"""
function backward!(v::AValue)
    topo = Any[]
    visited = IdSet{Any}()   # key on the (mutable) .grad array's identity

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
    fill!(v.grad, one(eltype(v.grad)))

    for node in reverse(topo)
        parent_grads = node.pullback_fn(node.grad)
        for (parent, parent_grad) in zip(node.parents, parent_grads)
            parent.grad .+= parent_grad
        end
    end
end