import Base: +, -, *, /, log, exp, sum

"""
    AValue{D<:AbstractArray, G<:AbstractArray, P<:Tuple, F}

Node in a vector-based automatic differentiation graph.

Each node stores its forward value `data`, accumulated gradient `grad`, 
its parent nodes `parents` and a pullback function `pullback_fn` that is 
used during reverse-mode automatic differentiation.
"""
struct AValue{D<:AbstractArray, G<:AbstractArray, P<:Tuple, F}
    data::D
    grad::G
    parents::P
    pullback_fn::F
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
function Value(val::T) where T
    return Value(val, zero(val), (), ())
end

# Add
+(a::Value, b::Value) = Value(a.data + b.data, zero(a.data + b.data), (a, b), (one(a.data), one(b.data)))
+(a::Value, b::Real) = a + Value(b)
+(a::Real, b::Value) = Value(a) + b

# Subract / negate
-(a::Value) = a * -1
-(a::Value, b::Value) = Value(a.data - b.data, zero(a.data - b.data), (a, b), (one(a.data), -one(b.data)))
-(a::Value, b::Real) = a - Value(b)
-(a::Real, b::Value) = Value(a) - b

# Multiply
*(a::Value, b::Value) = Value(a.data * b.data, zero(a.data * b.data), (a, b), (b.data, a.data))
*(a::Value, b::Real) = a * Value(b)
*(a::Real, b::Value) = Value(a) * b

# Divide
/(a::Value, b::Value) = Value(a.data / b.data, zero(a.data / b.data), (a, b), (one(a.data) / b.data, -a.data / b.data^2))
/(a::Value, b::Real) = a / Value(b)
/(a::Real, b::Value) = Value(a) / b

# Pow
^(a::Value, b::Real) = Value(a.data^b, zero(a.data^b), (a,), (b * (a.data^(b - one(b))),))

# Log
log(a::Value) = Value(log(a.data), zero(log(a.data)), (a,), (one(a.data) / a.data,))

# Exp
function exp(a::Value)
    out = exp(a.data)
    return Value(out, zero(out), (a,), (out,))
end

# ReLU
function relu(a::Value)
    out = a.data > zero(a.data) ? a.data : zero(a.data)
    local_grad = a.data > zero(a.data) ? one(a.data) : zero(a.data)
    return Value(out, zero(out), (a,), (local_grad,))
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
    visited = IdSet{Any}()

    function build_topo(node)
        if !(node in visited)
            push!(visited, node)
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