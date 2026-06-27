```@meta
CurrentModule = MicroGPT
```

# Optimizer

The optimizer updates the parameters of a model from the gradients computed by
the [Autograd](autograd.md) engine. MicroGPT provides [`Adam`](@ref), which
maintains per-parameter moment estimates across update steps. See the original
paper, [*Adam: A Method for Stochastic Optimization*](https://arxiv.org/abs/1412.6980),
for the underlying algorithm.

A training step has three parts: zero the gradients with [`zero_grad!`](@ref),
run a forward/backward pass to populate them (see [`backward!`](@ref)), and apply
one update with [`step!`](@ref).



```@example optimizer
using MicroGPT

# A parameter to optimize and a tiny "loss" that is minimal at W = 0.
W = AValue([1.0 2.0; 3.0 4.0])
opt = Adam([W]; α=0.1)

for _ in 1:50
    zero_grad!(opt)           # Set the gradients to zero
    loss = sum(relu(W * W))   # Define the loss
    backward!(loss)           # Backpropagation
    step!(opt)                # Apply an update step
end

W.data
```

Each call to [`step!`](@ref) advances the optimizer's timestep, which is used for
the bias correction of the moment estimates, so the moments accumulated across
iterations carry over between steps.

## Hyperparameters

[`Adam`](@ref) is constructed from a flat `Vector` of [`AValue`](@ref)
parameters and accepts the standard Adam hyperparameters as keyword arguments:

| Keyword | Default | Meaning                       |
|:--------|:--------|:------------------------------|
| `α`     | `0.01`  | learning rate                 |
| `β1`    | `0.85`  | first-moment decay            |
| `β2`    | `0.99`  | second-moment decay           |
| `ϵ`     | `1e-8`  | numerical stabilizer          |

```julia
opt = Adam(params; α=0.001, β1=0.9, β2=0.999)
```

## Reference

```@docs
Adam
step!
zero_grad!
```
