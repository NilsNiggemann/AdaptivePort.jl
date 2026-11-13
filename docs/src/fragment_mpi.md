## MPI.jl (`MPIExecutor`)

Use it when the job already runs under `mpiexec` (for example one Slurm allocation) and you want no extra worker
processes. All ranks run the same script; rank 0 runs the learner and the other ranks evaluate `f`.

```julia
using MPI, AdaptivePort
MPI.Init()
f(x) = expensive(x)                      # defined on every rank
learner = Learner1D(f; bounds = (-1, 1))
MPIExecutor(learner.f) do executor       # ranks > 0 serve f here and skip the block
    BlockingRunner(learner; tol = 1e-3, executor)
    save_data("result.csv", learner)     # rank 0 only
end
```

Run with `mpiexec -n 4 julia --project script.jl`; `ntasks` defaults to the number of ranks minus one.

- `MPIExecutor(body, f; comm = MPI.COMM_WORLD)` returns the value of `body` on rank 0 and `nothing` on the others.
  Code after the block runs on every rank, so guard rank-0 work with `MPI.Comm_rank(comm) == 0`.
- Every rank holds `f` itself; only points and results are sent (serialized Julia objects). The executor throws an
  `ArgumentError` if called with another function, so pass the learner's own `learner.f`.
- A throwing `f` is reported to rank 0 as an `EvaluationError` and the worker keeps serving.
- Leaving the block, normally or by an exception, sends a stop message to every worker. Workers finish the
  evaluation in progress first, so leaving takes as long as the slowest running evaluation; its result is dropped.
- At least 2 ranks are needed (`ArgumentError` otherwise). A crashed rank is not detected: the run hangs.
  All MPI calls run on the thread that called `MPI.Init`; call the executor from that thread.
