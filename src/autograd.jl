import Base: +, -, *, /, ^, log, exp

struct AValue{D<:AbstractArray, G<:AbstractArray, P<:Tuple, F}
    data::D
    grad::G
    parents::P
    pullback_fn::F
end

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

function relu(a::AValue)
    output = max.(zero(eltype(a.data)), a.data)

    return AValue(
        output,
        zero(output),
        (a,),
        dc -> (dc .* (a.data .> 0),)
    )
end


function backward!(v::AValue)
    topo = Any[]
    visited = IdSet{Value}()

    function build_topo(node::Value)
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

# TODO: Remove scalar-based autograd

"""
    Value{T,C<:Tuple,G<:Tuple}

A node in a scalar reverse-mode autograd graph.
"""
mutable struct Value{T,C<:Tuple,G<:Tuple}
    data::T
    grad::T
    children::C
    local_grads::G
end

"""
    Value(val)

Construct a leaf node holding `val` with zero gradient and no children.
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
    backward!(v::Value)

Compute the gradients of `v` with respect to every `Value` in its computation
graph via backpropagation.

This mutates the `v.grad` and of all `Value`s in its graph. Gradients
are accumulated, so reset them between independent backward passes if a
node is reused.
"""
function backward!(v::Value)
    topo = Value[]
    visited = Set{Value}()

    function build_topo(node::Value)
        if !(node in visited)
            push!(visited, node)
            for child in node.children
                build_topo(child)
            end
            push!(topo, node)
        end
    end
    build_topo(v)
    v.grad = one(v.data)

    for node in reverse(topo)
        for (child, local_grad) in zip(node.children, node.local_grads)
            child.grad += local_grad * node.grad  # Accumulate gradients from parents to children
        end
    end
end
