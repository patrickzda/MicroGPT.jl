using Test          # built-in testing library
using MicroGPT      # our own package
using ForwardDiff   # external comparison package

# Helper: run our function to get gradients
function our_grads(build, arrays::AbstractArray...)
    leaves = [AValue(copy(a)) for a in arrays]
    backward!(build(leaves...))
    return [copy(l.grad) for l in leaves]
end

# Helper: run ForwardDiff to get reference gradients
function fd_grads(build, arrays::AbstractArray...)
    sizes = size.(arrays)
    lengths = length.(arrays)
    offsets = cumsum([0; collect(lengths)])

    flat = vcat(vec.(arrays)...)
    g = ForwardDiff.gradient(flat) do z
        args = ntuple(length(arrays)) do i
            reshape(z[offsets[i]+1:offsets[i+1]], sizes[i])
        end
        sum(build(args...))
    end

    return [reshape(g[offsets[i]+1:offsets[i+1]], sizes[i]) for i in 1:length(arrays)]
end

@testset "autograd.jl" begin
    @testset "Leaf construction" begin
        a = AValue([1.0, 2.0, 3.0])
        @test a.data == [1.0, 2.0, 3.0]
        @test a.grad == zeros(3)
        @test a.parents == ()
        # The leaf pullback ignores its argument and returns no parent grads.
        @test a.pullback_fn(a.grad) == ()
    end

    # One testset per primitive operation, checking forward value and gradients against hardcoded expectations.
    @testset "Primitives" begin
        @testset "AValue + AValue" begin
            a = AValue([1.0, 2.0, 3.0])
            b = AValue([4.0, 5.0, 6.0])
            c = a + b
            backward!(c)
            @test c.data == [5.0, 7.0, 9.0]
            @test a.grad == [1.0, 1.0, 1.0]
            @test b.grad == [1.0, 1.0, 1.0]
            @test_throws DimensionMismatch a + AValue([1.0, 2.0])
        end

        @testset "AValue + Real & Real + AValue" begin
            a = AValue([1.0, 2.0, 3.0])
            c = a + 5.0
            backward!(c)
            @test c.data == [6.0, 7.0, 8.0]
            @test a.grad == [1.0, 1.0, 1.0]

            b = AValue([1.0, 2.0, 3.0])
            d = 5.0 + b
            backward!(d)
            @test d.data == [6.0, 7.0, 8.0]
            @test b.grad == [1.0, 1.0, 1.0]
        end

        @testset "AValue - AValue" begin
            a = AValue([5.0, 7.0, 9.0])
            b = AValue([4.0, 5.0, 6.0])
            c = a - b
            backward!(c)
            @test c.data == [1.0, 2.0, 3.0]
            @test a.grad == [1.0, 1.0, 1.0]
            @test b.grad == [-1.0, -1.0, -1.0]
            @test_throws DimensionMismatch a - AValue([1.0, 2.0])
        end

        @testset "AValue - Real & Real - AValue" begin
            a = AValue([5.0, 7.0, 9.0])
            c = a - 2.0
            backward!(c)
            @test c.data == [3.0, 5.0, 7.0]
            @test a.grad == [1.0, 1.0, 1.0]

            b = AValue([5.0, 7.0, 9.0])
            d = 10.0 - b
            backward!(d)
            @test d.data == [5.0, 3.0, 1.0]
            @test b.grad == [-1.0, -1.0, -1.0]
        end

        @testset "- (negation)" begin
            a = AValue([1.0, -2.0, 3.0])
            c = -a
            backward!(c)
            @test c.data == [-1.0, 2.0, -3.0]
            @test a.grad == [-1.0, -1.0, -1.0]
        end

        @testset "Matrix * Matrix" begin
            a = AValue([1.0 2.0; 3.0 4.0])
            b = AValue([5.0 6.0; 7.0 8.0])
            c = a * b
            backward!(c)
            @test c.data == [1.0 2.0; 3.0 4.0] * [5.0 6.0; 7.0 8.0]
            @test a.grad == ones(2, 2) * transpose([5.0 6.0; 7.0 8.0])
            @test b.grad == transpose([1.0 2.0; 3.0 4.0]) * ones(2, 2)
            @test_throws DimensionMismatch a * AValue([1.0 2.0 3.0])
        end

        @testset "Matrix * Vector" begin
            a = AValue([1.0 2.0; 3.0 4.0])
            b = AValue([5.0, 6.0])
            c = a * b
            backward!(c)
            @test c.data == [1.0 2.0; 3.0 4.0] * [5.0, 6.0]
            @test a.grad == ones(2) * transpose([5.0, 6.0])
            @test b.grad == transpose([1.0 2.0; 3.0 4.0]) * ones(2)
            @test_throws DimensionMismatch a * AValue([1.0, 2.0, 3.0])
        end

        @testset "AValue * Real & Real * AValue" begin
            a = AValue([1.0, 2.0, 3.0])
            c = a * 3.0
            backward!(c)
            @test c.data == [3.0, 6.0, 9.0]
            @test a.grad == [3.0, 3.0, 3.0]

            b = AValue([1.0, 2.0, 3.0])
            d = 3.0 * b
            backward!(d)
            @test d.data == [3.0, 6.0, 9.0]
            @test b.grad == [3.0, 3.0, 3.0]
        end

        @testset "mul_elementwise" begin
            a = AValue([1.0, 2.0, 3.0])
            b = AValue([4.0, 5.0, 6.0])
            c = mul_elementwise(a, b)
            backward!(c)
            @test c.data == [4.0, 10.0, 18.0]
            @test a.grad == [4.0, 5.0, 6.0]
            @test b.grad == [1.0, 2.0, 3.0]
            @test_throws DimensionMismatch mul_elementwise(a, AValue([1.0, 2.0]))
        end

        @testset "AValue / Real" begin
            a = AValue([2.0, 4.0, 6.0])
            c = a / 2.0
            backward!(c)
            @test c.data == [1.0, 2.0, 3.0]
            @test a.grad == [0.5, 0.5, 0.5]
        end

        @testset "div_elementwise" begin
            a = AValue([1.0, 2.0, 3.0])
            b = AValue([2.0, 4.0, 6.0])
            c = div_elementwise(a, b)
            backward!(c)
            @test c.data ≈ [0.5, 0.5, 0.5]
            @test a.grad ≈ [1.0, 1.0, 1.0] ./ [2.0, 4.0, 6.0]
            @test b.grad ≈ -[1.0, 2.0, 3.0] ./ ([2.0, 4.0, 6.0] .^ 2)
            @test_throws DimensionMismatch div_elementwise(a, AValue([1.0, 2.0]))
        end

        @testset "pow_elementwise_scalar" begin
            a = AValue([1.0, 2.0, 3.0])
            c = pow_elementwise_scalar(a, 3)
            backward!(c)
            @test c.data == [1.0, 8.0, 27.0]
            @test a.grad == 3 .* ([1.0, 2.0, 3.0] .^ 2)
        end

        @testset "log" begin
            a = AValue([1.0, exp(1.0), exp(2.0)])
            c = log(a)
            backward!(c)
            @test c.data ≈ [0.0, 1.0, 2.0]
            @test a.grad ≈ 1.0 ./ [1.0, exp(1.0), exp(2.0)]
        end

        @testset "exp" begin
            a = AValue([0.0, 1.0, 2.0])
            c = exp(a)
            backward!(c)
            @test c.data ≈ exp.([0.0, 1.0, 2.0])
            @test a.grad ≈ exp.([0.0, 1.0, 2.0])
        end

        @testset "sum" begin
            a = AValue([1.0, 2.0, 3.0])
            c = sum(a)
            @test c.data == fill(6.0)
            backward!(c)
            @test a.grad == [1.0, 1.0, 1.0]
        end

        @testset "relu" begin
            a = AValue([-2.0, 0.0, 3.0])
            c = relu(a)
            backward!(c)
            @test c.data == [0.0, 0.0, 3.0]
            @test a.grad == [0.0, 0.0, 1.0]
        end
    end

    # Verify that gradients accumulate correctly when a node is reused in the computation graph.
    @testset "Gradient accumulation" begin
        @testset "a + a" begin
            a = AValue([1.0, 2.0, 3.0])
            c = a + a
            backward!(c)
            @test c.data == [2.0, 4.0, 6.0]
            @test a.grad == [2.0, 2.0, 2.0]
        end

        @testset "mul_elementwise(a, a)" begin
            a = AValue([1.0, 2.0, 3.0])
            c = mul_elementwise(a, a)
            backward!(c)
            @test c.data == [1.0, 4.0, 9.0]
            @test a.grad == 2 .* [1.0, 2.0, 3.0]
        end
    end

    # Cross-check gradients of composite expressions against ForwardDiff to verify correct chain rule implementation.
    @testset "ForwardDiff comparison" begin
        @testset "sum(exp(x) + log(x))" begin
            x = [1.0, 2.0, 3.0]
            build_a(v) = exp(v) + log(v)
            build_p(v) = exp.(v) .+ log.(v)
            ours = our_grads(build_a, x)
            refs = fd_grads(build_p, x)
            @test ours[1] ≈ refs[1]
        end

        @testset "sum(relu(x) + pow(x, 3))" begin
            x = [-1.0, 2.0, -0.5, 3.0]
            build_a(v) = relu(v) + pow_elementwise_scalar(v, 3)
            build_p(v) = max.(0.0, v) .+ v .^ 3
            ours = our_grads(build_a, x)
            refs = fd_grads(build_p, x)
            @test ours[1] ≈ refs[1]
        end

        @testset "sum(div_elementwise(x, y) - y * 2)" begin
            x = [1.0, 2.0, 3.0]
            y = [2.0, 4.0, 8.0]
            build_a(u, v) = div_elementwise(u, v) - v * 2.0
            build_p(u, v) = u ./ v .- v .* 2.0
            ours = our_grads(build_a, x, y)
            refs = fd_grads(build_p, x, y)
            @test ours[1] ≈ refs[1]
            @test ours[2] ≈ refs[2]
        end

        @testset "sum(A * B)" begin
            A = [1.0 2.0; 3.0 4.0]
            B = [5.0 6.0; 7.0 8.0]
            build(a, b) = a * b
            ours = our_grads(build, A, B)
            refs = fd_grads(build, A, B)
            @test ours[1] ≈ refs[1]
            @test ours[2] ≈ refs[2]
        end

        @testset "sum(A * x) with matrix-vector product" begin
            A = [1.0 2.0; 3.0 4.0]
            x = [5.0, 6.0]
            build(a, v) = a * v
            ours = our_grads(build, A, x)
            refs = fd_grads(build, A, x)
            @test ours[1] ≈ refs[1]
            @test ours[2] ≈ refs[2]
        end
    end

    # Verify gradient correctness for larger expressions combining multiple operations and reused nodes.
    @testset "Composite expression a*b + a" begin
        a = AValue([1.0 2.0; 3.0 4.0])
        b = AValue([2.0 0.0; 1.0 3.0])
        build(x, y) = x * y + x
        c = build(a, b)
        backward!(c)
        @test c.data == [1.0 2.0; 3.0 4.0] * [2.0 0.0; 1.0 3.0] .+ [1.0 2.0; 3.0 4.0]

        ours = our_grads(build, [1.0 2.0; 3.0 4.0], [2.0 0.0; 1.0 3.0])
        refs = fd_grads(build, [1.0 2.0; 3.0 4.0], [2.0 0.0; 1.0 3.0])
        @test ours[1] ≈ refs[1]
        @test ours[2] ≈ refs[2]
    end

    # Higher-level layers built on top of the primitives.
    @testset "Layers" begin
        @testset "linear" begin
            x = [1.0, 2.0, -1.0]
            W = [1.0 0.0 2.0; 0.0 1.0 1.0]
            xv, Wv = AValue(x), AValue(W)
            y = linear(xv, Wv)
            backward!(y)
            @test y.data == W * x
            @test xv.grad == transpose(W) * ones(2)
            @test Wv.grad == ones(2) * transpose(x)

            build_a(a, b) = linear(a, b)
            build_p(a, b) = b * a
            ours = our_grads(build_a, x, W)
            refs = fd_grads(build_p, x, W)
            @test ours[1] ≈ refs[1]
            @test ours[2] ≈ refs[2]
        end

        @testset "softmax" begin
            x = [1.0, 2.0, 3.0]
            p = softmax(AValue(x))
            # Softmax is shift-invariant, so it matches the naive normalised exponential.
            @test p.data ≈ exp.(x) ./ sum(exp.(x))
            @test sum(p.data) ≈ 1.0

            build_a(v) = pow_elementwise_scalar(softmax(v), 2)
            function build_p(v)
                e = exp.(v .- maximum(v))
                (e ./ sum(e)) .^ 2
            end
            ours = our_grads(build_a, x)
            refs = fd_grads(build_p, x)
            @test ours[1] ≈ refs[1]
        end

        @testset "rmsnorm" begin
            x = [1.0, 2.0, 3.0, 4.0]
            eps = 1e-5
            y = rmsnorm(AValue(x); eps = eps)
            scale = (sum(x .^ 2) / length(x) + eps) ^ -0.5
            @test y.data ≈ x .* scale

            build_a(v) = rmsnorm(v) # uses default
            function build_p(v)
                n = length(v)
                s = (sum(v .^ 2) / n + 1e-5) ^ -0.5
                v .* s
            end
            ours = our_grads(build_a, x)
            refs = fd_grads(build_p, x)
            @test ours[1] ≈ refs[1]
        end
    end
end
