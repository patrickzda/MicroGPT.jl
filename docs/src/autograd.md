```@meta
CurrentModule = MicroGPT
```

# Autograd

Wraps matrices / vectors in [`AValue`](@ref), builds an expression, and calls [`backward!`](@ref)
to populate the gradients of every node in the graph:

```@example autograd
using MicroGPT

W = AValue([1.0 2.0; 3.0 4.0])
x = AValue([5.0, 6.0])

y = W * x
loss = sum(relu(y))
backward!(loss)
```
