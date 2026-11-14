# Parallel execution

The `executor` keyword of a runner decides where the learner's function runs; `ntasks` decides how many
evaluations run at once. The default is `ThreadedExecutor` with `ntasks = 1`, so a default run is serial. The learner itself always stays in the calling process and is never touched
concurrently: only the runner loop calls `ask` and `tell!`.

| Executor | Runs `f` on | Load | `ntasks` default |
|:---------|:------------|:-----|:-----------------|
| [`ThreadedExecutor`](@ref) (default) | threads of this process | built in | 1 |
| [`DistributedExecutor`](@ref) | Distributed.jl worker processes, on one or many machines | `using Distributed` | number of workers |
| [`DaggerExecutor`](@ref) | whatever Dagger's scheduler picks (threads or workers) | `using Dagger` | Dagger's processors |
| [`MPIExecutor`](@ref) | MPI ranks of an `mpiexec` job | `using MPI` | ranks − 1 |

!!! tip "Which executor should I use?"
    - One machine, thread-safe `f`: threads (`julia -t auto`, `ntasks = Threads.nthreads()`).
    - `f` not thread-safe, or several machines: `DistributedExecutor`.
    - Already inside an `mpiexec`/`srun` MPI job, or `f` itself uses MPI: `MPIExecutor`.
    - Already using Dagger: `DaggerExecutor`.

!!! note "Parallel speed-up needs an expensive function"
    The learner runs in one process. The runner tells all results that have arrived and then asks for all free
    slots in one `ask` call, so the learner's cost is shared by a batch: `Learner1D` costs about 0.1 ms per point,
    and `Learner2D` one triangulation per `ask` (about 7 ms at 1000 points), plus one per `loss` check when
    `tol` is set. The workers stay busy as long as an evaluation takes much longer than one such round. The
    real price of many workers is placement: points of one batch are chosen without each other's values, so the
    same number of points reaches a somewhat higher loss. Measured speed-ups are on the
    [home page](../index.md#Performance).

## Distributed.jl, MPI or Dagger

| | Distributed.jl | MPI | Dagger |
|:--|:--|:--|:--|
| Setup | `SlurmClusterManager.jl` starts the workers with `srun` | MPI.jl configured for the cluster's MPI | Distributed workers, registered with Dagger |
| Functions on workers | define with `@everywhere` | every rank runs the whole script | named functions with `@everywhere` |
| A worker crashes | the run stops with an [`EvaluationError`](@ref) | the same if the rank exits or the launcher ends the job (`srun --kill-on-bad-exit=1`); a rank stuck in `f` needs `timeout` | as Dagger reports it; not tested |
| Interactive use | add workers to a running REPL | batch jobs only | add workers to a running REPL |
| Overhead per evaluation | one remote call | one message each way | a scheduler round on top of a remote call |

Use Distributed.jl unless your cluster or your `f` already relies on MPI. Dagger schedules graphs of dependent
tasks; the learner already decides what to evaluate next and its evaluations are independent, so Dagger adds
overhead and pays off only when `f` itself is a Dagger computation.

## Threads

```julia
using AdaptivePort                      # julia -t 4
learner = Learner1D(f; bounds = (-1, 1))
BlockingRunner(learner; tol = 0.01, ntasks = Threads.nthreads())
```

!!! warning "Thread safety of `f`"
    With `ntasks > 1`, `f` runs on several threads at once. It must not write to shared buffers or global state
    without a lock, and must use its own random number generator.

## Distributed.jl workers

[`DistributedExecutor`](@ref) evaluates the learner's function on Distributed.jl worker processes (`using
Distributed` loads the extension). Use it when `f` is expensive enough that sending a point and a value is
negligible, on one machine or a cluster.

```julia
using Distributed, AdaptivePort
addprocs(4)
@everywhere using AdaptivePort
@everywhere f(x) = tanh(50x)          # f and everything it uses must exist on the workers
learner = Learner1D(f; bounds = (-1, 1))
BlockingRunner(learner; tol = 0.01, executor = DistributedExecutor())
```

!!! warning "Define `f` on every worker"
    Only points and values travel. Define `f` and its dependencies with `@everywhere` (or load them in the worker
    startup), *after* adding the workers. Otherwise the first evaluation fails with an `UndefVarError` on the
    worker.

- The default pool is `CachingPool(workers())`, which ships a closure to each worker once instead of on every
  call. `ntasks` defaults to the pool size, so every worker evaluates one point at a time.
- Workers from any cluster manager work unchanged, for example `addprocs(SlurmManager())` from
  SlurmClusterManager.jl inside a Slurm allocation (see [Clusters](clusters.md)).
- An exception in `f` is reported as an [`EvaluationError`](@ref) with the original exception and the worker's
  backtrace. A lost worker also ends the run with an `EvaluationError`; there are no retries.

`examples/distributed.jl` is a runnable script.

## MPI.jl

[`MPIExecutor`](@ref) is for jobs that already run under `mpiexec` or `srun` and should start no extra
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

Run it with `mpiexec -n 4 julia --project script.jl` (`examples/mpi.jl`), using the `mpiexec` that matches your
MPI library. With MPI.jl's bundled MPICH:

```julia
using MPI
MPI.mpiexec(m -> run(`$m -n 4 julia --project=examples examples/mpi.jl`))
```

`ntasks` defaults to the number of ranks minus one.

!!! warning "Crashed ranks: set `timeout` on clusters"
    A rank that leaves early on its own (`exit` inside `f`, an error outside `f`) tells rank 0, and its point
    fails with an [`EvaluationError`](@ref). A rank killed by a signal (segfault, out of memory) cannot say
    anything: MPI.jl's bundled `mpiexec` then ends the whole job, but `srun` without `--kill-on-bad-exit`, and
    a rank stuck inside `f`, leave rank 0 waiting. Under Slurm use `srun --kill-on-bad-exit=1`. Pass `timeout` (seconds, longer than any normal evaluation)
    to fail such a point and stop using its rank. Use `checkpoint` as well, so that a killed job loses little.
    A timed-out rank that is still stuck keeps the MPI job alive after rank 0 is done; end the script with
    `MPI.Abort(MPI.COMM_WORLD, 1)` on rank 0 if a warning reports timed-out ranks.

- `MPIExecutor(body, f; comm = MPI.COMM_WORLD, timeout = Inf)` returns the value of `body` on rank 0 and `nothing` on the
  others. Code after the block runs on every rank, so guard rank-0 work with `MPI.Comm_rank(comm) == 0`.
- Every rank holds `f` itself; only points and results are sent (serialized Julia objects). The executor throws
  an `ArgumentError` if called with another function, so pass the learner's own `learner.f`.
- A throwing `f` is reported to rank 0 as an `EvaluationError`, and the worker keeps serving. A lost or
  timed-out rank gets no further points; once every rank is lost, all queued points fail. A timed-out rank whose
  result arrives late is used again.
- Leaving the block, normally or by an exception, sends a stop message to every worker. Workers finish their
  current evaluation first, so leaving takes as long as the slowest running evaluation; its result is dropped.
- At least 2 ranks are needed (`ArgumentError` otherwise). All MPI calls run on the thread that called
  `MPI.Init`; call the executor from that thread.

## Dagger.jl

[`DaggerExecutor`](@ref) sends every evaluation to Dagger's scheduler (`Dagger.spawn(f, options, x...)`), which
places it on a thread or worker process it knows. Use it when you already use Dagger, or want threads and
Distributed workers behind one executor.

```julia
using Distributed, AdaptivePort
addprocs(2)
@everywhere using Dagger, AdaptivePort
@everywhere f(x) = tanh(50x)             # a named function, not a closure
using Dagger
Dagger.addprocs!(Dagger.Sch.eager_context(), workers())
learner = Learner1D(f; bounds = (-1, 1))
BlockingRunner(learner; tol = 1e-2, executor = DaggerExecutor())
```

!!! warning "Register workers with Dagger"
    Dagger does not notice workers added after its scheduler context was created. Call
    `Dagger.addprocs!(Dagger.Sch.eager_context(), workers())` (or add workers before first use); otherwise
    tasks fail with a `SchedulingException`.

- Keywords go to `Dagger.Options`, e.g. `DaggerExecutor(; scope = Dagger.ProcessScope(2))` restricts evaluation
  to worker 2. `default_ntasks` is `Dagger.num_processors(scope)`.
- `f` must be a named function defined on every worker that may run it (`@everywhere`). Closures are sent
  with every evaluation and intermittently fail with a `MethodError` on multi-threaded workers.
- A throwing `f` yields an `EvaluationError` holding the original exception and its backtrace. There are no
  retries.

`examples/dagger.jl` is a runnable script.

## Advanced: custom executors

An executor is any callable `executor(f, x)` that starts the evaluation and returns a handle that `fetch` can
wait on (a `Task`, a `Distributed.Future`, a Dagger thunk, ...); `f` is the learner's function and `x` the point
(splatted for tuple points). `fetch` must throw if `f` threw. Optionally define
`AdaptivePort.default_ntasks(executor)` ([`default_ntasks`](@ref)), the number of evaluations to keep in flight.
The runner needs nothing else, so a custom executor is a few lines:

```julia
struct SpawnExecutor end                                  # runs every call in its own task
(::SpawnExecutor)(f, x) = Threads.@spawn f(x...)
AdaptivePort.default_ntasks(::SpawnExecutor) = Threads.nthreads()
BlockingRunner(learner; tol = 0.01, executor = SpawnExecutor())
```

A handle whose `fetch` wraps the original exception in its own type (as `RemoteException` does) can define
`AdaptivePort.captured_exception` so that [`EvaluationError`](@ref) shows the original error.
