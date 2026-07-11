import Pkg

Pkg.activate(@__DIR__)
Pkg.add(
    [
        "BenchmarkTools",
        "CUDA",
        "DiffEqGPU",
        "ForwardDiff",
        "JSON3",
        "OrdinaryDiffEqLowOrderRK",
        "OrdinaryDiffEqNonlinearSolve",
        "OrdinaryDiffEqRosenbrock",
        "OrdinaryDiffEqTsit5",
        "ReverseDiff",
        "SciMLBase",
        "SciMLSensitivity",
        "StaticArrays",
        "StochasticDiffEq",
        "Zygote",
    ]
)
for (package, version) in [
        "BenchmarkTools" => "1",
        "CUDA" => "6",
        "DiffEqGPU" => "3",
        "ForwardDiff" => "1",
        "JSON3" => "1",
        "OrdinaryDiffEqLowOrderRK" => "2",
        "OrdinaryDiffEqNonlinearSolve" => "2",
        "OrdinaryDiffEqRosenbrock" => "2",
        "OrdinaryDiffEqTsit5" => "2",
        "ReverseDiff" => "1",
        "SciMLBase" => "3",
        "SciMLSensitivity" => "7",
        "StaticArrays" => "1",
        "StochasticDiffEq" => "7",
        "Zygote" => "0.7",
        "julia" => "1.12",
    ]
    Pkg.compat(package, version)
end
Pkg.precompile()
