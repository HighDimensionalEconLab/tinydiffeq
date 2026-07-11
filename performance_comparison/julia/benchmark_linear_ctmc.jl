using BenchmarkTools
using ExponentialUtilities
using JSON3
using LinearAlgebra
using Statistics
using Zygote

struct RingGenerator{T, V <: AbstractVector{T}}
    rates::V
end

struct AdjointRingGenerator{T, V <: AbstractVector{T}}
    rates::V
end

Base.eltype(::RingGenerator{T}) where {T} = T
Base.eltype(::AdjointRingGenerator{T}) where {T} = T
Base.size(operator::Union{RingGenerator, AdjointRingGenerator}) =
    (length(operator.rates), length(operator.rates))
Base.size(operator::Union{RingGenerator, AdjointRingGenerator}, dimension::Int) =
    size(operator)[dimension]
Base.adjoint(operator::RingGenerator) = AdjointRingGenerator(operator.rates)
LinearAlgebra.ishermitian(::Union{RingGenerator, AdjointRingGenerator}) = false
LinearAlgebra.opnorm(
    operator::Union{RingGenerator, AdjointRingGenerator}, norm_order::Real = Inf
) = 2 * maximum(operator.rates)

function LinearAlgebra.mul!(
        output::AbstractVector,
        operator::RingGenerator,
        input::AbstractVector,
    )
    rates = operator.rates
    last_index = length(rates)
    @inbounds output[1] = rates[last_index] * input[last_index] - rates[1] * input[1]
    @inbounds for index in 2:last_index
        output[index] = rates[index - 1] * input[index - 1] - rates[index] * input[index]
    end
    return output
end

function LinearAlgebra.mul!(
        output::AbstractVector,
        operator::AdjointRingGenerator,
        input::AbstractVector,
    )
    rates = operator.rates
    last_index = length(rates)
    @inbounds for index in 1:(last_index - 1)
        output[index] = rates[index] * (input[index + 1] - input[index])
    end
    @inbounds output[last_index] = rates[last_index] * (input[1] - input[last_index])
    return output
end

function propagate(operator, initial, horizon, krylov_dim, tolerance, num_substeps)
    state = initial
    substep = horizon / num_substeps
    for _ in 1:num_substeps
        state = expv(
            substep,
            operator,
            state;
            m = krylov_dim,
            tol = tolerance,
            opnorm = opnorm(operator),
            ishermitian = false,
        )
    end
    return state
end

function propagate_adaptive(operator, initial, horizon, krylov_dim, tolerance)
    return expv_timestep(
        horizon,
        operator,
        initial;
        adaptive = true,
        m = krylov_dim,
        tol = tolerance,
        opnorm = opnorm(operator),
        ishermitian = false,
        NA = 2 * length(initial),
    )
end

function adaptive_value_and_vjp(
        operator,
        initial,
        cotangent,
        horizon,
        krylov_dim,
        tolerance,
    )
    value = propagate_adaptive(
        operator, initial, horizon, krylov_dim, tolerance
    )
    gradient = propagate_adaptive(
        operator', cotangent, horizon, krylov_dim, tolerance
    )
    return value, gradient
end

function manual_value_and_vjp(
        operator,
        initial,
        cotangent,
        horizon,
        krylov_dim,
        tolerance,
        num_substeps,
    )
    value = propagate(
        operator, initial, horizon, krylov_dim, tolerance, num_substeps
    )
    gradient = propagate(
        operator', cotangent, horizon, krylov_dim, tolerance, num_substeps
    )
    return value, gradient
end

function make_problem(::Type{T}, num_states::Int) where {T}
    indices = T.(0:(num_states - 1))
    denominator = T(max(num_states - 1, 1))
    rates = T(0.1) .+ T(0.9) .* indices ./ denominator
    initial = one(T) .+ sin.(T(0.017) .* indices) .^ 2
    initial ./= sum(initial)
    cotangent = cos.(T(0.013) .* indices) .+ T(0.2) .* sin.(T(0.007) .* indices)
    return RingGenerator(rates), initial, cotangent
end

function timing(function_to_measure; quick::Bool)
    function_to_measure()
    sample_count = quick ? 20 : 100
    seconds = quick ? 0.25 : 1.0
    trial = @benchmark $function_to_measure() samples = sample_count seconds = seconds
    estimate = median(trial)
    return Dict(
        "median_seconds" => estimate.time * 1.0e-9,
        "memory_bytes" => estimate.memory,
        "allocations" => estimate.allocs,
    )
end

function zygote_status(
        operator,
        initial,
        cotangent,
        horizon,
        krylov_dim,
        tolerance,
        num_substeps,
    )
    objective(state) = dot(
        cotangent,
        propagate(operator, state, horizon, krylov_dim, tolerance, num_substeps),
    )
    try
        gradient = Zygote.gradient(objective, initial)[1]
        return all(isfinite, gradient) ? "supported" : "nonfinite"
    catch error
        return "unsupported: $(typeof(error))"
    end
end

function main()
    output_path = ARGS[1]
    quick = "--quick" in ARGS
    sizes = quick ? (10_000, 100_000) : (10_000, 100_000, 1_000_000)
    horizon = 10.0
    krylov_dim = 30
    num_substeps = 2
    results = Dict{String, Any}[]
    for data_type in (Float32, Float64)
        tolerance = data_type === Float32 ? 1.0e-5 : 1.0e-10
        for num_states in sizes
            operator, initial, cotangent = make_problem(data_type, num_states)
            primal_function() = propagate(
                operator,
                initial,
                data_type(horizon),
                krylov_dim,
                data_type(tolerance),
                num_substeps,
            )
            vjp_function() = manual_value_and_vjp(
                operator,
                initial,
                cotangent,
                data_type(horizon),
                krylov_dim,
                data_type(tolerance),
                num_substeps,
            )
            adaptive_primal_function() = propagate_adaptive(
                operator,
                initial,
                data_type(horizon),
                krylov_dim,
                data_type(tolerance),
            )
            adaptive_vjp_function() = adaptive_value_and_vjp(
                operator,
                initial,
                cotangent,
                data_type(horizon),
                krylov_dim,
                data_type(tolerance),
            )
            primal_value = primal_function()
            _, vjp_value = vjp_function()
            primal_timing = timing(primal_function; quick = quick)
            value_and_vjp_timing = timing(vjp_function; quick = quick)
            adaptive_primal_timing = timing(
                adaptive_primal_function; quick = quick
            )
            adaptive_vjp_timing = timing(adaptive_vjp_function; quick = quick)
            push!(
                results,
                Dict(
                    "states" => num_states,
                    "dtype" => string(data_type),
                    "horizon" => horizon,
                    "krylov_dim" => krylov_dim,
                    "num_substeps" => num_substeps,
                    "tolerance" => tolerance,
                    "primal" => primal_timing,
                    "value_and_vjp_manual" => value_and_vjp_timing,
                    "adaptive_primal" => adaptive_primal_timing,
                    "adaptive_value_and_vjp_manual" => adaptive_vjp_timing,
                    "adaptive_policy" => "Niesen-Wright adaptive tau and m",
                    "mass_error" => abs(sum(primal_value) - one(data_type)),
                    "objective" => dot(cotangent, primal_value),
                    "vjp_norm" => norm(vjp_value),
                    "zygote" => zygote_status(
                        operator,
                        initial,
                        cotangent,
                        data_type(horizon),
                        krylov_dim,
                        data_type(tolerance),
                        num_substeps,
                    ),
                ),
            )
            println(
                "SciML ",
                data_type,
                " N=",
                num_states,
                " primal=",
                round(1.0e6 * primal_timing["median_seconds"], digits = 2),
                " us value+VJP=",
                round(1.0e6 * value_and_vjp_timing["median_seconds"], digits = 2),
                " us",
            )
        end
    end
    payload = Dict(
        "metadata" => Dict(
            "julia" => string(VERSION),
            "threads" => Threads.nthreads(),
            "blas_threads" => BLAS.get_num_threads(),
            "quick" => quick,
        ),
        "results" => results,
    )
    open(output_path, "w") do io
        JSON3.pretty(io, payload)
    end
    return nothing
end

main()
