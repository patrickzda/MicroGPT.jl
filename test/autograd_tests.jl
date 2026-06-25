using Test          # built-in testing library
using MicroGPT      # our own package
using ForwardDiff   # external comparison package

# Helper: run our function to get gradients
function our_grad(f, vals::Float64...) # vals::Float64...-> accept any number of Float64 arguments
    vs = Value.(vals)
    backward!(f(vs...))
    return [v.grad for v in vs]
end

# Helper: run ForwardDiff to get reference gradients
fd_grad(f, vals::Float64...) = ForwardDiff.gradient(x -> f(x...), collect(vals)) # x -> f(x...) -> vector unpacked to seperate args

@testset "autograd.jl" begin

    @testset "a*b + a" begin
        a = Value(2.0)
        b = Value(3.0)
        c = a * b + a
        backward!(c)
        @test c.data == 8.0
        @test a.grad == 4.0
        @test b.grad == 2.0
    end

    # One testset per primitive operation -> checking by fixed values
    @testset "Primitives" begin

        @testset "+" begin
            a, b = Value(2.0), Value(3.0)
            backward!(a + b)
            @test a.grad == 1.0
            @test b.grad == 1.0
        end

        @testset "- (subtraction operation)" begin
            a, b = Value(5.0), Value(3.0)
            backward!(a - b)
            @test a.grad == 1.0
            @test b.grad == -1.0
        end

        @testset "- (negative sign)" begin
            a = Value(3.0)
            backward!(-a)
            @test a.grad == -1.0
        end

        @testset "*" begin
            a, b = Value(2.0), Value(3.0)
            backward!(a * b)
            @test a.grad == 3.0
            @test b.grad == 2.0
        end

        @testset "/" begin
            a, b = Value(6.0), Value(2.0)
            backward!(a / b)
            @test a.grad == 0.5
            @test b.grad == -1.5
        end

        @testset "^" begin
            a = Value(3.0)
            backward!(a^3)
            @test a.grad == 27.0
        end

        @testset "log" begin
            a = Value(exp(1.0))
            backward!(log(a))
            @test a.grad ≈ exp(-1.0)
        end

        @testset "exp" begin
            a = Value(2.0)
            backward!(exp(a))
            @test a.grad ≈ exp(2.0)
        end

        @testset "relu (positive input)" begin
            a = Value(3.0)
            backward!(relu(a))
            @test a.grad == 1.0
        end

        @testset "relu (negative input)" begin
            a = Value(-2.0)
            backward!(relu(a))
            @test a.grad == 0.0
        end

    end

    # Checking if it would overwrite .grad instead of adding to it
    @testset "Gradient accumulation (a*a)" begin
        a = Value(4.0)
        backward!(a * a)
        @test a.grad == 8.0
    end

    # Cross check against ForwardDiff -> Pass same arguments -> If they match, everything is correct
    @testset "ForwardDiff comparison" begin

        @testset "log(exp(a) + b)" begin
            f(a, b) = log(exp(a) + b)
            @test our_grad(f, 1.0, 2.0) ≈ fd_grad(f, 1.0, 2.0)
        end

        @testset "(a + b)^3 / b" begin
            f(a, b) = (a + b)^3 / b
            @test our_grad(f, 1.0, 2.0) ≈ fd_grad(f, 1.0, 2.0)
        end

        @testset "a*b - log(a) + exp(b)" begin
            f(a, b) = a * b - log(a) + exp(b)
            @test our_grad(f, 2.0, 1.0) ≈ fd_grad(f, 2.0, 1.0)
        end

    end

end
