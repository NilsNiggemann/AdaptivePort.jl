## Distributed.jl workers

`DistributedExecutor` evaluates the learner's function on Distributed.jl worker processes (`using Distributed`
loads the extension). Use it when `f` is expensive enough that sending a point and a value is negligible, on one
machine or a cluster.

```julia
using Distributed, AdaptivePort
addprocs(4)
@everywhere using AdaptivePort
@everywhere f(x) = tanh(50x)          # f and everything it uses must exist on the workers
learner = Learner1D(f; bounds = (-1, 1))
BlockingRunner(learner; tol = 0.01, executor = DistributedExecutor())
```

- The default pool is `CachingPool(workers())`, which ships a closure to each worker once instead of on every call.
  `ntasks` defaults to the pool size, so every worker evaluates one point at a time. A call takes a free worker
  from the pool and returns it when the call has finished.
- Only points and values travel. Define `f` and its dependencies with `@everywhere` (or load them in the
  worker startup).
- Clusters: `addprocs(SlurmManager(n))` (ClusterManagers.jl) or MPIClusterManagers.jl work unchanged.
- An exception in `f` is reported as an `EvaluationError` with the original exception and the worker's backtrace.
  A lost worker also ends the run with an `EvaluationError`; there are no retries.
