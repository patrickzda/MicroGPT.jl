using Test
using MicroGPT

# This test has been implemented with AI assistance (Claude)

# Independent, plain-array reference implementation of Adam to test with fixed gradients
# No use of any AD
function adam_reference(θ0, grads; α, β1, β2, ϵ)
    θ = copy(θ0)
    m = zero(θ)
    v = zero(θ)
    for (t, g) in enumerate(grads)
        m .= β1 .* m .+ (1 - β1) .* g
        v .= β2 .* v .+ (1 - β2) .* g .^ 2
        m̂ = m ./ (1 - β1^t)
        v̂ = v ./ (1 - β2^t)
        θ .-= α .* m̂ ./ (sqrt.(v̂) .+ ϵ)
    end
    return θ
end

# Helper: set the .grad of each parameter from a plain gradient vector.
set_grads!(params, g) =
    for (p, gi) in zip(params, g)
        ;
        p.grad = gi;
    end

@testset "optimizer.jl" begin
    # Adam optimizes array-valued `AValue` parameters. We cross check against the
    # same reference, treating each array entry as an independent scalar parameter.
    # AValue is an immutable struct, so its .grad must be set in place.
    @testset "Adam matches reference implementation" begin
        θ0 = [1.0, -2.0, 0.5]
        grads = [[0.30, -0.10, 0.20],
            [0.25, -0.15, 0.10],
            [-0.05, 0.20, -0.30],
            [0.40, 0.05, 0.15]]
        hp = (α=0.01, β1=0.85, β2=0.99, ϵ=1e-8)

        p = AValue(copy(θ0))
        opt = Adam([p]; hp...)

        for g in grads
            p.grad .= g
            step!(opt)
        end

        expected = adam_reference(θ0, grads; hp...)
        @test p.data ≈ expected
        @test opt.t == length(grads)
    end

    # Same closed-form first-step check as for scalar Adam, but per array element.
    @testset "Adam first step ≈ α·sign(grad)" begin
        p = AValue([5.0, -3.0])
        opt = Adam([p]; α=0.01, β1=0.85, β2=0.99, ϵ=1e-8)
        p.grad .= [2.0, -4.0]
        step!(opt)
        @test p.data[1] ≈ 5.0 - 0.01    # positive grad -> step down
        @test p.data[2] ≈ -3.0 + 0.01   # negative grad -> step up
    end

    # Test the reset of the gradients
    @testset "zero_grad! clears the gradients (Adam)"  begin
        p = AValue([5.0, -3.0])
        opt = Adam([p]; α=0.01, β1=0.85, β2=0.99, ϵ=1e-8)
        p.grad .= [2.0, -4.0]
        @test p.grad[1] == 2.0
        @test p.grad[2] == -4.0
        zero_grad!(opt)
        @test all(p.grad == zeros(size(p.grad)))
    end

end
