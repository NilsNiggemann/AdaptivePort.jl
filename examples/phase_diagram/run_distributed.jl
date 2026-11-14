# Phase diagram on several nodes with Distributed.jl workers.
# Locally (4 worker processes):  julia --project=examples examples/phase_diagram/run_distributed.jl
# On a Slurm cluster (4 nodes):  sbatch examples/phase_diagram/job_distributed.sh
# Rerunning continues from the saved data.
using Distributed, SlurmClusterManager, AdaptivePort

if haskey(ENV, "SLURM_JOB_ID")
    addprocs(SlurmManager())  # one worker per Slurm task, on all nodes of the allocation
else
    addprocs(4)
end

@everywhere using AdaptivePort
@everywhere include($(joinpath(@__DIR__, "model.jl")))
@everywhere f(g, T) = order_parameters(g, T; cost = 0.05)  # f must exist on every worker

path = joinpath(@__DIR__, "phase_diagram.csv")
learner = Learner2D(f; bounds = ((0.0, 1.0), (0.0, 1.0)), vdim = 2)  # (g, T) -> (m_A, m_B)
isfile(path) && load_data!(learner, path)  # resume an earlier run

BlockingRunner(learner; max_npoints = 1500, duration = 110 * 60, executor = DistributedExecutor(), checkpoint = path, checkpoint_interval = 60)
println("$(npoints(learner)) points on $(nworkers()) workers, loss $(round(loss(learner); sigdigits = 3)), saved to $path")
