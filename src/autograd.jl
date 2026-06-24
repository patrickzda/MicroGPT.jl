import Base: +, -, *, /, ^, log, exp

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
function Value(val::T) where {T<:AbstractFloat}
    return Value(val, zero(typeof(val)), (), ())
end

"""
    Value(val::Real)

Construct a leaf node from any real number, converting `data` and `grad` to floats.
Just for easy definitions
"""
function Value(val::Real)
    return Value(float(val))
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
