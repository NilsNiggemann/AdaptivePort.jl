# Evaluate a learner's function on Distributed.jl worker processes.
# Run with: julia --project=. examples/distributed.jl
using Distributed, AdaptivePort

addprocs(2)
@everywhere begin
    using AdaptivePort
    slow_f(x) = (sleep(0.05); tanh(50x))  # defined on every worker; only points and values travel
end

learner = Learner1D(slow_f; bounds = (-1, 1))
executor = DistributedExecutor()  # CachingPool(workers()); ntasks defaults to the number of workers
t = @elapsed BlockingRunner(learner; max_npoints = 100, executor)
println("$(npoints(learner)) points on $(nworkers()) workers in $(round(t; digits = 2)) s, loss $(loss(learner))")
rmprocs(workers())
