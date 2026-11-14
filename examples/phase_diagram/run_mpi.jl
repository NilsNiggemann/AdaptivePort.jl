# Phase diagram on several nodes with MPI: rank 0 runs the learner, all other ranks evaluate.
# Locally (MPI.jl's bundled MPICH, 5 ranks = 4 evaluators):
#   julia --project=examples -e 'using MPI; MPI.mpiexec(m -> run(`$m -n 5 julia --project=examples examples/phase_diagram/run_mpi.jl`))'
# On a Slurm cluster (4 nodes):  sbatch examples/phase_diagram/job_mpi.sh
using MPI, AdaptivePort
MPI.Init()

include(joinpath(@__DIR__, "model.jl"))      # every rank runs this file, so f exists on every rank
f(g, T) = order_parameters(g, T; cost = 0.05)

path = joinpath(@__DIR__, "phase_diagram.csv")
learner = Learner2D(f; bounds = ((0.0, 1.0), (0.0, 1.0)), vdim = 2)

MPIExecutor(learner.f) do executor           # ranks > 0 serve f here and skip the block
    isfile(path) && load_data!(learner, path)
    BlockingRunner(learner; max_npoints = 1500, duration = 110 * 60, executor, checkpoint = path, checkpoint_interval = 60)
    println("$(npoints(learner)) points on $(default_ntasks(executor)) evaluating ranks, saved to $path")
end
