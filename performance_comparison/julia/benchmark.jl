using BenchmarkTools: @benchmark
using ForwardDiff
using JSON3
using LinearAlgebra: BLAS, Diagonal
import SciMLBase
import SciMLSensitivity
import StochasticDiffEq
using OrdinaryDiffEqLowOrderRK: Euler, RK4
using OrdinaryDiffEqRosenbrock: Rodas5P
using OrdinaryDiffEqTsit5: Tsit5
using SciMLBase: EnsembleProblem, EnsembleSerial, ODEFunction, ODEProblem, SDEFunction,
    SDEProblem, remake, solve
using SciMLSensitivity
using Statistics: median, quantile
using StochasticDiffEq: EM, ImplicitEM
using Zygote

BLAS.set_num_threads(1)

function parse_args(args)
    quick = "--quick" in args
    subset_index = findfirst(==("--subset"), args)
    subset = isnothing(subset_index) ? nothing : args[subset_index + 1]
    transform_index = findfirst(==("--transform"), args)
    transform = isnothing(transform_index) ? nothing : args[transform_index + 1]
    method_index = findfirst(==("--method"), args)
    method = isnothing(method_index) ? nothing : args[method_index + 1]
    output_index = findfirst(==("--output"), args)
    if isnothing(output_index) || output_index == length(args)
        error("--output PATH is required")
    end
    return (; quick, subset, transform, method, output = args[output_index + 1])
end

function measurement(run, argument, quick)
    started = time_ns()
    value = run(argument)
    warmup_seconds = (time_ns() - started) / 1.0e9
    sample_count = quick ? 7 : 40
    duration = quick ? 0.2 : 2.0
    trial = @benchmark $run($argument) samples = sample_count seconds = duration
    times = sort!(Float64.(trial.times) ./ 1.0e9)
    return Dict(
        "median_seconds" => median(times),
        "iqr_seconds" => quantile(times, 0.75) - quantile(times, 0.25),
        "samples_seconds" => times,
        "iterations_per_sample" => 1,
        "compile_seconds" => warmup_seconds,
        "first_execute_seconds" => warmup_seconds,
        "allocations" => Int(median(trial).allocs),
        "memory_bytes" => Int(median(trial).memory),
        "warmup_value_type" => string(typeof(value)),
    )
end

function ode_data(equation, ::Type{T}) where {T}
    if equation == "ode_scalar"
        return (;
            initial = one(T), tspan = (zero(T), T(10)), n_steps = 64,
            dt_adaptive = T(0.1), max_steps = 512,
        )
    end
    n = 256
    initial = [sin(T(2pi) * T(i - 1) / T(n)) for i in 1:n]
    return (;
        initial, tspan = (zero(T), one(T)), n_steps = 128,
        dt_adaptive = T(0.01), max_steps = 512,
    )
end

scalar_field(u, p, t) = p * u

function vector_field!(du, u, p, t)
    n = length(u)
    @inbounds for i in eachindex(u)
        left = i == 1 ? u[n] : u[i - 1]
        right = i == n ? u[1] : u[i + 1]
        du[i] = p * u[i] + typeof(p)(0.1) * (left - 2 * u[i] + right)
    end
    return nothing
end

function make_ode_run(
        equation, method, stepping, transform, ::Type{T}, rtol, atol
    ) where {T}
    data = ode_data(equation, T)
    problem = if equation == "ode_scalar"
        if transform == "vjp"
            field_vjp(u, p, t) = p[1] * u
            ODEProblem(field_vjp, data.initial, data.tspan, T[-0.2])
        else
            field_primal(u, p, t) = T(-0.2) * u
            ODEProblem(field_primal, data.initial, data.tspan)
        end
    else
        if transform == "vjp"
            function field_vjp!(du, u, p, t)
                return vector_field!(du, u, p[1], t)
            end
            ODEProblem(field_vjp!, data.initial, data.tspan, T[-0.2])
        else
            function field_primal!(du, u, p, t)
                return vector_field!(du, u, T(-0.2), t)
            end
            ODEProblem(field_primal!, data.initial, data.tspan)
        end
    end
    algorithm = if method == "euler"
        Euler()
    elseif method == "rk4"
        RK4()
    elseif method == "rodas5p"
        Rodas5P()
    else
        Tsit5()
    end
    adaptive = stepping == "adaptive"
    dt = adaptive ? data.dt_adaptive : (data.tspan[2] - data.tspan[1]) / data.n_steps

    function run(initial)
        configured = remake(problem; u0 = initial)
        common = (;
            adaptive,
            dt,
            reltol = T(rtol),
            abstol = T(atol),
            maxiters = adaptive ? data.max_steps : data.n_steps,
            save_everystep = false,
            save_start = false,
            save_end = true,
            dense = false,
            calck = false,
        )
        solution = if transform == "vjp"
            solve(
                configured, algorithm; common...,
                sensealg = SciMLSensitivity.ReverseDiffAdjoint()
            )
        else
            solve(configured, algorithm; common...)
        end
        return solution.u[end]
    end
    return run, data.initial, dt, adaptive ? data.max_steps : data.n_steps
end

function make_dae_run(size, stepping, transform, ::Type{T}, rtol, atol) where {T}
    y0 = size == 1 ? T[1] : collect(range(T(0.8), T(1.2); length = size))
    u0 = vcat(y0, y0)
    mass = Diagonal(vcat(ones(T, size), zeros(T, size)))
    residual!, parameters = if transform == "vjp"
        function residual_with_parameters!(du, u, p, t)
            y = @view u[1:size]
            z = @view u[(size + 1):(2size)]
            @inbounds for i in 1:size
                du[i] = p[1] * z[i]
                du[size + i] = z[i] + T(0.1) * (z[i]^3 - y[i]^3) - y[i]
            end
            return nothing
        end
        residual_with_parameters!, T[-0.2]
    else
        function residual_without_parameters!(du, u, p, t)
            y = @view u[1:size]
            z = @view u[(size + 1):(2size)]
            @inbounds for i in 1:size
                du[i] = T(-0.2) * z[i]
                du[size + i] = z[i] + T(0.1) * (z[i]^3 - y[i]^3) - y[i]
            end
            return nothing
        end
        residual_without_parameters!, nothing
    end
    function_object = ODEFunction(residual!; mass_matrix = mass)
    problem = ODEProblem(function_object, u0, (zero(T), one(T)), parameters)
    adaptive = stepping == "adaptive"
    dt = adaptive ? T(0.05) : T(1 / 64)
    max_steps = adaptive ? 256 : 64
    function run(initial)
        configured = remake(problem; u0 = initial)
        common = (;
            adaptive, dt, reltol = T(rtol), abstol = T(atol), maxiters = max_steps,
            save_everystep = false, save_start = false, save_end = true,
            dense = false, calck = false,
        )
        solution = if transform == "vjp"
            solve(
                configured, Rodas5P(); common...,
                sensealg = SciMLSensitivity.ReverseDiffAdjoint()
            )
        else
            solve(configured, Rodas5P(); common...)
        end
        return solution.u[end]
    end
    return run, u0, dt, max_steps
end

function make_sde_run(batch, ::Type{T}) where {T}
    initial = collect(range(T(0.8), T(1.2); length = batch))
    drift(u, p, t) = p[1] * u
    diffusion(u, p, t) = p[2] * u
    problem = SDEProblem(drift, diffusion, one(T), (zero(T), one(T)), T[-0.2, 0.1])
    function problem_function(prob, context)
        return remake(prob; u0 = initial[context.sim_id])
    end
    ensemble = EnsembleProblem(problem; prob_func = problem_function, safetycopy = false)
    function run(values)
        solution = solve(
            ensemble, EM(), EnsembleSerial(); trajectories = length(values),
            dt = T(1 / 128), adaptive = false, seed = 1729,
            save_everystep = false, save_start = false, save_end = true,
            dense = false
        )
        return [trajectory.u[end] for trajectory in solution.u]
    end
    return run, initial, T(1 / 128), 128
end

function make_sdae_run(batch, ::Type{T}) where {T}
    initial = collect(range(T(0.8), T(1.2); length = batch))
    mass = Diagonal(T[1, 0])
    function drift!(du, u, p, t)
        y, z = u
        du[1] = T(-0.2) * z
        du[2] = z + T(0.1) * (z^3 - y^3) - y
        return nothing
    end
    function diffusion!(du, u, p, t)
        du[1] = T(0.1) * u[2]
        du[2] = zero(T)
        return nothing
    end
    function_object = SDEFunction(drift!, diffusion!; mass_matrix = mass)
    problem = SDEProblem(function_object, T[1, 1], (zero(T), one(T)))
    function problem_function(prob, context)
        value = initial[context.sim_id]
        return remake(prob; u0 = T[value, value])
    end
    ensemble = EnsembleProblem(problem; prob_func = problem_function, safetycopy = false)
    function run(values)
        solution = solve(
            ensemble, ImplicitEM(), EnsembleSerial();
            trajectories = length(values), dt = T(1 / 128), adaptive = false,
            seed = 1729, save_everystep = false, save_start = false,
            save_end = true, dense = false
        )
        return [trajectory.u[end] for trajectory in solution.u]
    end
    return run, initial, T(1 / 128), 128
end

function transformed(run, initial, transform)
    if transform == "primal"
        return run
    elseif transform == "jvp"
        direction = similar(initial)
        fill!(direction, one(eltype(direction)))
        return value -> ForwardDiff.derivative(
            epsilon -> run(value .+ epsilon .* direction), zero(eltype(value))
        )
    end
    return value -> Zygote.withgradient(x -> sum(run(x)), value)
end

function scalar_transformed(run, initial, transform)
    if transform == "primal"
        return run
    elseif transform == "jvp"
        return value -> ForwardDiff.derivative(run, value)
    end
    return value -> Zygote.withgradient(run, value)
end

function record_case(
        library, equation, size, method, stepping, controller, dtype_name,
        transform_name, batch, rtol, atol, quick
    )
    T = dtype_name == "float32" ? Float32 : Float64
    if startswith(equation, "ode")
        run, initial, dt, max_steps = make_ode_run(
            equation, method, stepping, transform_name, T, rtol, atol
        )
        timed = initial isa Number ?
            scalar_transformed(run, initial, transform_name) :
            transformed(run, initial, transform_name)
    elseif startswith(equation, "dae")
        if method != "rodas5p"
            error("SciML exact explicit root-restoring DAE method unavailable")
        end
        run, initial, dt, max_steps = make_dae_run(
            size, stepping, transform_name, T, rtol, atol
        )
        timed = transformed(run, initial, transform_name)
    elseif equation == "sde_ensemble"
        if transform_name != "primal"
            error("SciML pathwise ensemble derivative not enabled in this suite")
        end
        run, initial, dt, max_steps = make_sde_run(batch, T)
        timed = run
    elseif equation == "sdae_ensemble"
        if transform_name != "primal"
            error("SciML pathwise SDAE ensemble derivative not enabled in this suite")
        end
        run, initial, dt, max_steps = make_sdae_run(batch, T)
        timed = run
    else
        error("SciML exact explicit projected SDAE method unavailable")
    end
    inferred = try
        Core.Compiler.return_type(timed, Tuple{typeof(initial)}) !== Any
    catch
        false
    end
    result = measurement(timed, initial, quick)
    merge!(
        result, Dict(
            "library" => library,
            "case" => Dict(
                "equation" => equation,
                "size" => size,
                "method" => method,
                "stepping" => stepping,
                "controller" => controller,
                "dtype" => dtype_name,
                "transform" => transform_name,
                "batch" => batch,
            ),
            "rtol" => rtol,
            "atol" => atol,
            "dt_0" => Float64(dt),
            "max_steps" => max_steps,
            "stats" => Dict("ok" => true),
            "type_inferred" => inferred,
            "absolute_error" => nothing,
        )
    )
    return result
end

function case_specs(quick)
    dtypes = ["float32", "float64"]
    transforms = ["primal", "jvp", "vjp"]
    cases = NamedTuple[]
    for dtype in dtypes, transform in transforms
        for (equation, size) in [("ode_scalar", 1), ("ode_vector", 256)]
            for method in ["euler", "rk4", "tsit5", "rodas5p"]
                push!(
                    cases, (;
                        equation, size, method, stepping = "fixed",
                        controller = "none", dtype, transform, batch = 1,
                    )
                )
            end
            push!(
                cases, (;
                    equation, size, method = "tsit5", stepping = "adaptive",
                    controller = "native", dtype, transform, batch = 1,
                )
            )
            push!(
                cases, (;
                    equation, size, method = "rodas5p", stepping = "adaptive",
                    controller = "native", dtype, transform, batch = 1,
                )
            )
        end
        for size in [1, 32]
            equation = size == 1 ? "dae_scalar" : "dae_vector"
            push!(
                cases, (;
                    equation, size, method = "rodas5p", stepping = "fixed",
                    controller = "none", dtype, transform, batch = 1,
                )
            )
            push!(
                cases, (;
                    equation, size, method = "rodas5p", stepping = "adaptive",
                    controller = "native", dtype, transform, batch = 1,
                )
            )
        end
        for batch in [1, 256, 16384]
            push!(
                cases, (;
                    equation = "sde_ensemble", size = 1, method = "em",
                    stepping = "fixed", controller = "none", dtype, transform, batch,
                )
            )
            if !quick || batch <= 256
                push!(
                    cases, (;
                        equation = "sdae_ensemble", size = 1,
                        method = "implicit_em", stepping = "fixed",
                        controller = "native", dtype, transform, batch,
                    )
                )
            end
        end
    end
    return cases
end

function main()
    args = parse_args(ARGS)
    results = Any[]
    unavailable = Any[]
    cases = case_specs(args.quick)
    if !isnothing(args.subset)
        cases = filter(
            case -> case.equation == args.subset ||
                (args.subset in ("ode", "dae") && startswith(case.equation, args.subset)),
            cases
        )
    end
    if !isnothing(args.transform)
        cases = filter(case -> case.transform == args.transform, cases)
    end
    if !isnothing(args.method)
        cases = filter(case -> case.method == args.method, cases)
    end
    for case in cases
        tolerance_rows = case.dtype == "float32" ?
            [(1.0e-4, 1.0e-6, "common")] :
            [(1.0e-4, 1.0e-6, "common"), (1.0e-7, 1.0e-9, "precision")]
        for (rtol, atol, tolerance_name) in tolerance_rows
            if case.stepping != "adaptive" && tolerance_name != "common"
                continue
            end
            try
                result = record_case(
                    "sciml", case.equation, case.size, case.method,
                    case.stepping, case.controller, case.dtype, case.transform,
                    case.batch, rtol, atol, args.quick
                )
                result["tolerance_name"] = tolerance_name
                push!(results, result)
                println(
                    rpad(case.equation, 14), " ", rpad(case.method, 8), " ",
                    rpad(case.transform, 6), " ", rpad(case.dtype, 7), " b=",
                    rpad(case.batch, 5), " ", round(
                        result["median_seconds"] * 1.0e6;
                        digits = 2
                    ), " us"
                )
            catch exception
                push!(
                    unavailable, Dict(
                        "library" => "sciml",
                        "case" => Dict(string(key) => getfield(case, key) for key in keys(case)),
                        "rtol" => rtol,
                        "atol" => atol,
                        "reason" => sprint(showerror, exception),
                    )
                )
            end
        end
    end
    metadata = Dict(
        "backend" => "cpu",
        "julia" => string(VERSION),
        "packages" => Dict(
            "SciMLBase" => string(Base.pkgversion(SciMLBase)),
            "SciMLSensitivity" => string(Base.pkgversion(SciMLSensitivity)),
            "StochasticDiffEq" => string(Base.pkgversion(StochasticDiffEq)),
        ),
        "threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(),
        "quick" => args.quick,
        "platform" => Sys.MACHINE,
    )
    return open(args.output, "w") do io
        JSON3.pretty(
            io, Dict(
                "metadata" => metadata,
                "controller_equivalence" => Any[],
                "results" => results,
                "unavailable" => unavailable,
            )
        )
        write(io, '\n')
    end
end

main()
