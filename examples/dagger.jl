# Learn a function through Dagger's scheduler, on the threads of this process and on extra worker processes.
# Run with: julia --project=. -t 4 examples/dagger.jl   (needs Dagger in the active environment)
using Distributed, AdaptivePort
addprocs(2)
@everywhere using Dagger
using Dagger
Dagger.addprocs!(Dagger.Sch.eager_context(), workers())   # tell the running scheduler about workers added after it started

@everywhere f(x) = (sleep(0.01); tanh(50x))
learner = Learner1D(f; bounds = (-1, 1))
executor = DaggerExecutor()
println("Dagger processors: ", default_ntasks(executor))
BlockingRunner(learner; tol = 1e-2, executor)
println("points: ", npoints(learner), ", loss: ", loss(learner))
rmprocs(workers())
