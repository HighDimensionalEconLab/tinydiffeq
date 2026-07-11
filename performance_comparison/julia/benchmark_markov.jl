using BenchmarkTools
using JSON3
using JumpProcesses
using Random
using SciMLBase
using Statistics
using Test

function generator_matrix(num_states::Int, ::Type{T}) where {T <: AbstractFloat}
    rng = Xoshiro(num_states + 10_000)
    off_diagonal = rand(rng, T, num_states, num_states)
    for index in 1:num_states
        off_diagonal[index, index] = zero(T)
        off_diagonal[index, :] ./= sum(off_diagonal[index, :])
        off_diagonal[index, :] .*= T(0.5) + T(1.5) * rand(rng, T)
    end
    generator = copy(off_diagonal)
    for index in 1:num_states
        generator[index, index] = -sum(off_diagonal[index, :])
    end
    return generator
end

function jump_problem(generator::Matrix{T}, horizon::T, seed::Int) where {T}
    num_states = size(generator, 1)
    problem = DiscreteProblem(1, (zero(T), horizon))
    jumps = map(Iterators.product(1:num_states, 1:num_states)) do (source, target)
        rate = (state, parameters, time) ->
        state == source ? generator[source, target] : zero(T)
        affect! = integrator -> (integrator.u = target)
        ConstantRateJump(rate, affect!)
    end
    selected = Tuple(
        jumps[source, target]
            for source in 1:num_states for target in 1:num_states if source != target
    )
    return JumpProblem(problem, Direct(), selected...; rng = Xoshiro(seed), save_positions = (false, false))
end

function simulate_batch(problems::AbstractVector, horizon)
    endpoints = Vector{Int}(undef, length(problems))
    for trajectory in eachindex(problems)
        problem = problems[trajectory]
        solution = solve(problem, SSAStepper(); saveat = horizon)
        endpoints[trajectory] = solution.u[end]
    end
    return endpoints
end

function benchmark_case(num_states, length, batch, ::Type{T}; quick = false) where {T}
    generator = generator_matrix(num_states, T)
    horizon = T(length / 2)
    problems = [jump_problem(generator, horizon, trajectory) for trajectory in 1:batch]
    simulate_batch(problems[1:min(batch, 2)], horizon)
    @assert @inferred(simulate_batch(problems[1:1], horizon)) isa Vector{Int}
    benchmark = @benchmark simulate_batch($problems, $horizon) samples = 9 evals = 1
    estimate = median(benchmark)
    return Dict(
        "case" => Dict(
            "kind" => "continuous",
            "states" => num_states,
            "length" => length,
            "batch" => batch,
            "dtype" => T == Float32 ? "float32" : "float64",
            "method" => "sciml_direct_ssa",
        ),
        "median_seconds" => estimate.time / 1.0e9,
        "memory_bytes" => estimate.memory,
        "allocations" => estimate.allocs,
        "type_stable" => true,
    )
end

function main()
    output = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "../results/julia_cpu_markov.json")
    quick = "--quick" in ARGS
    selected = quick ?
        [(2, 64, 256)] :
        [(2, length, batch) for length in (64, 256) for batch in (1, 256)]
    results = [
        benchmark_case(states, length, batch, dtype; quick = quick)
            for (states, length, batch) in selected for dtype in (Float32, Float64)
    ]
    payload = Dict(
        "metadata" => Dict(
            "backend" => "cpu",
            "julia" => string(VERSION),
            "threads" => Threads.nthreads(),
            "quick" => quick,
            "implementation" => "JumpProcesses.Direct + SSAStepper",
        ),
        "results" => results,
    )
    mkpath(dirname(output))
    return open(output, "w") do stream
        JSON3.pretty(stream, payload)
    end
end

main()
