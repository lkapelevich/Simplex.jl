using Test
using JuMP
using GLPK
using LinearAlgebra
include(joinpath(dirname(@__DIR__()), "src", "readable", "algorithm.jl"))

function test_unboundedness(A, b, c, primal_sol, dual_sol, obj)
    tol = 1e-10
    @test obj == -Inf
    @test dot(c, primal_sol) < 0
    @test A * primal_sol ≈ zeros(size(A, 1)) atol=tol rtol=tol
    @test all(primal_sol .>= -tol)

    model = Model(with_optimizer(GLPK.Optimizer))
    @variable(model, x[1:length(c)] >= 0)
    @constraint(model, constr, A * x .== b)
    @objective(model, Min, dot(c, x))
    optimize!(model)
    @test termination_status(model) == MOI.DUAL_INFEASIBLE
end

function test_optimality(A, b, c, primal_sol, dual_sol, obj)
    tol = 1e-10
    @test isfinite(obj)
    @test A * primal_sol ≈ b
    @test all(primal_sol .> -tol)
    @test dot(c, primal_sol) ≈ dot(b, dual_sol) ≈ obj
    @test all(c - A' * dual_sol .>= -tol)
    @test dot(primal_sol, c - A' * dual_sol) ≈ 0 atol=tol rtol=tol

    model = Model(with_optimizer(GLPK.Optimizer))
    @variable(model, x[1:length(c)] >= 0)
    @constraint(model, constr, A * x .== b)
    @objective(model, Min, dot(c, x))
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test objective_value(model) ≈ obj
end

function test_infeasibility(A, b, c, primal_sol, dual_sol, obj)
    @test obj == Inf

    model = Model(with_optimizer(GLPK.Optimizer))
    @variable(model, x[1:length(c)] >= 0)
    @constraint(model, constr, A * x .== b)
    @objective(model, Min, dot(c, x))
    optimize!(model)
    @test termination_status(model) == MOI.INFEASIBLE
end

# normal
@testset "problem 1" begin
    A = Float64[1 -1 1 1 0 0; 1 1 0 0 -1 0; 0 0 1 0 0 1]
    b = Float64[4, 0, 6]
    c = Float64[1, 2, 3, 0, 0, 0]
    (primal_sol, dual_sol, obj) = fullrsm(A, b, c)
    test_optimality(A, b, c, primal_sol, dual_sol, obj)
end

# normal
@testset "problem 2" begin
    A = Float64[3 2 1 2 1 0 0; 1 1 1 1 0 1 0; 4 3 3 4 0 0 1]
    b = Float64[225, 117, 420]
    c = -Float64[19, 13, 12, 17, 0, 0, 0]
    (primal_sol, dual_sol, obj) = fullrsm(A, b, c)
    test_optimality(A, b, c, primal_sol, dual_sol, obj)
end

# normal
@testset "problem 3" begin
    A = Float64[3 2 1 2 1 0 0; 1 1 1 1 0 1 0; 4 3 3 4 0 0 1]
    b = Float64[225, 117, 420]
    c = -Float64[19, 13, 12, 17, 0, 0, 0]
    (primal_sol, dual_sol, obj) = fullrsm(A, b, c)
    test_optimality(A, b, c, primal_sol, dual_sol, obj)
end

# unbounded
@testset "problem 4" begin
    A = Float64[0 1 0; -1 0 1]
    b = Float64[1, 1]
    c = Float64[1, 2, -3]
    (primal_sol, dual_sol, obj) = fullrsm(A, b, c)
    test_unboundedness(A, b, c, primal_sol, dual_sol, obj)
end

# infeasible
@testset "problem 5" begin
    A = Matrix{Float64}(I, 3, 2)
    A[3, :] = [1, 1]
    b = Float64[1, 1, 1]
    c = Float64[1, 2]
    (primal_sol, dual_sol, obj) = fullrsm(A, b, c)
    test_infeasibility(A, b, c, primal_sol, dual_sol, obj)
end

# infeasible
@testset "problem 6" begin
    c = Float64[-1, -1, -1, -1, 0, 0]
    A = Float64[1 3 2 4 1 0; 3 1 2 1 0 1; 5 3 3 3 0 0]
    b = Float64[5, 4, 9]
    (primal_sol, dual_sol, obj) = fullrsm(A, b, c)
    test_infeasibility(A, b, c, primal_sol, dual_sol, obj)
end

# infeasible
@testset "problem 7" begin
     c = Float64[1, 1, 0, 0, 0]
     A = Float64[1 0 -1 0 0; 0 1 0 -1 0; 1 1 0 0 1]
     b = Float64[6, 6, 11]
    (primal_sol, dual_sol, obj) = fullrsm(A, b, c)
    test_infeasibility(A, b, c, primal_sol, dual_sol, obj)
end

# unbounded
@testset "problem 8" begin
    c = Float64[-1, -1, 0, 0]
    A = Float64[1 -1 -1 0; 1 1 0 -1]
    b = Float64[1, 2]
    (primal_sol, dual_sol, obj) = fullrsm(A, b, c)
    test_unboundedness(A, b, c, primal_sol, dual_sol, obj)
end

 # unbounded
 @testset "problem 9" begin
    c = Float64[0, -2, -1, 0, 0, 0]
    A = Float64[1 -1 0 1 0 0; -2 1 0 0 1 0; 0 1 -2 0 0 1]
    b = Float64[5, 3, 5];
    (primal_sol, dual_sol, obj) = fullrsm(A, b, c)
    test_unboundedness(A, b, c, primal_sol, dual_sol, obj)
end
