# Adaptive sampling on MPI ranks. Run with
#   mpiexec -n 4 julia --project=examples examples/mpi.jl
# with the mpiexec of the MPI library MPI.jl uses; for its bundled MPICH:
#   julia --project=examples -e 'using MPI; MPI.mpiexec(m -> run(`$m -n 4 julia --project=examples examples/mpi.jl`))'
# Rank 0 runs the learner; ranks 1..3 evaluate.
using MPI, AdaptivePort
MPI.Init()

f(x) = (sleep(0.05); tanh(50x))   # defined on every rank

learner = Learner1D(f; bounds = (-1.0, 1.0))
MPIExecutor(learner.f) do executor
    BlockingRunner(learner; tol = 1e-3, executor)
    println("npoints = ", npoints(learner), ", loss = ", loss(learner), " on ", default_ntasks(executor), " workers")
end
