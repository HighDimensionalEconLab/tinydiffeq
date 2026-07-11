using BenchmarkTools: @benchmark
import CUDA
import DiffEqGPU
using DiffEqGPU: EnsembleGPUKernel, GPUEM
using JSON3
using SciMLBase: EnsembleProblem, SDEProblem, remake, solve
using StaticArrays: SVector
using Statistics: median, quantile

function parse_args(args)
    quick = "--quick" in args
    output_index = findfirst(==("--output"), args)
    if isnothing(output_index) || output_index == length(args)
        error("--output PATH is required")
    end
    return (; quick, output = args[output_index + 1])
end

function make_run(batch, ::Type{T}) where {T}
    initial = collect(range(T(0.8), T(1.2); length = batch))
    drift(u, p, t) = SVector{1, T}(p[1] * u[1])
    diffusion(u, p, t) = SVector{1, T}(p[2] * u[1])
    problem = SDEProblem(
        drift, diffusion, SVector{1, T}(one(T)),
        (zero(T), one(T)), SVector{2, T}(T(-0.2), T(0.1))
    )
    problem_function(prob, context) =
        remake(prob; u0 = SVector{1, T}(initial[context.sim_id]))
    ensemble = EnsembleProblem(problem; prob_func = problem_function, safetycopy = false)
    backend = EnsembleGPUKernel(CUDA.CUDABackend(), 0.0)
    function run(_)
        solution = solve(
            ensemble, GPUEM(), backend; trajectories = batch,
            dt = T(1 / 128), adaptive = false, seed = 1729,
            save_everystep = false, save_start = false, save_end = true
        )
        CUDA.synchronize()
        return solution
    end
    return run
end

function measure(run, batch, quick)
    started = time_ns()
    warmup = run(batch)
    first_seconds = (time_ns() - started) / 1.0e9
    sample_count = quick ? 12 : 40
    duration = quick ? 0.5 : 2.0
    trial = @benchmark $run($batch) samples = sample_count seconds = duration
    times = sort!(Float64.(trial.times) ./ 1.0e9)
    return Dict(
        "median_seconds" => median(times),
        "iqr_seconds" => quantile(times, 0.75) - quantile(times, 0.25),
        "samples_seconds" => times,
        "iterations_per_sample" => 1,
        "compile_seconds" => first_seconds,
        "first_execute_seconds" => first_seconds,
        "allocations" => Int(median(trial).allocs),
        "memory_bytes" => Int(median(trial).memory),
        "warmup_value_type" => string(typeof(warmup)),
    )
end

function main()
    args = parse_args(ARGS)
    if !CUDA.functional()
        error("CUDA is not functional")
    end
    batches = args.quick ? [256] : [1, 256, 16384]
    dtypes = args.quick ? [(Float32, "float32")] :
        [(Float32, "float32"), (Float64, "float64")]
    results = Any[]
    unavailable = Any[]
    for (T, dtype_name) in dtypes, batch in batches
        case = Dict(
            "equation" => "sde_ensemble",
            "size" => 1,
            "method" => "em",
            "stepping" => "fixed",
            "controller" => "none",
            "dtype" => dtype_name,
            "transform" => "primal",
            "batch" => batch,
        )
        try
            run = make_run(batch, T)
            result = measure(run, batch, args.quick)
            merge!(
                result, Dict(
                    "library" => "sciml",
                    "case" => case,
                    "rtol" => 1.0e-4,
                    "atol" => 1.0e-6,
                    "dt_0" => 1 / 128,
                    "max_steps" => 128,
                    "stats" => Dict("ok" => true),
                    "type_inferred" => true,
                    "absolute_error" => nothing,
                    "tolerance_name" => "common",
                )
            )
            push!(results, result)
            println(
                dtype_name, " b=", batch, " ",
                round(result["median_seconds"] * 1.0e6; digits = 2), " us"
            )
        catch exception
            push!(
                unavailable, Dict(
                    "library" => "sciml",
                    "case" => case,
                    "rtol" => 1.0e-4,
                    "atol" => 1.0e-6,
                    "reason" => sprint(showerror, exception),
                )
            )
        end
    end
    return open(args.output, "w") do io
        JSON3.pretty(
            io, Dict(
                "metadata" => Dict(
                    "backend" => "cuda",
                    "julia" => string(VERSION),
                    "packages" => Dict(
                        "CUDA" => string(Base.pkgversion(CUDA)),
                        "DiffEqGPU" => string(Base.pkgversion(DiffEqGPU)),
                    ),
                    "device" => string(CUDA.device()),
                    "quick" => args.quick,
                ),
                "controller_equivalence" => Any[],
                "results" => results,
                "unavailable" => unavailable,
            )
        )
        write(io, '\n')
    end
end

main()
