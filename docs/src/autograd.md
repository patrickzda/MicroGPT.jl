```@meta
CurrentModule = MicroGPT
```

# Autograd

Wraps matrices / vectors in [`AValue`](@ref), builds an expression, and calls [`backward!`](@ref)
to populate the gradients of every node in the graph:

```@example autograd
using MicroGPT

a = AValue([1.0 2.0])
b = AValue([3.0 4.0])
c = a * b + relu(a - b)
backward!(c)

(value = c.data, da = a.grad, db = b.grad)
```
