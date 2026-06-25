```@meta
CurrentModule = MicroGPT
```

# Autograd

Wrap scalars in [`Value`](@ref), build an expression, and call [`backward!`](@ref)
to populate the gradients of every node in the graph:

```@example autograd
using MicroGPT

a = Value(2.0)
b = Value(3.0)
c = a * b + relu(a - b)
backward!(c)

(value = c.data, da = a.grad, db = b.grad)
```
