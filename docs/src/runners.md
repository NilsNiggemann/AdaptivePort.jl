# Runners and parallel execution


A runner does the `ask` → evaluate → `tell!` loop for you, using the function stored in the
learner (`f(x)` for [`Learner1D`](@ref), `f(x, y)` for [`Learner2D`](@ref)): construct a learner,
construct a runner, wait, then read the result. Learners decide *where* to sample; runners decide
*when and how* to evaluate. The manual loop above stays fully supported.

```julia
learner = Learner1D(f; bounds=(-1, 1))
runner = BlockingRunner(learner; tol=0.02, max_npoints=500)   # returns when done
data = result(learner)

runner = AsyncRunner(learner; tol=0.01)   # returns immediately
# ... other work ...
wait(runner)
```

`examples/runners.jl` shows blocking, asynchronous and parallel runs
(`julia -t 4 --project=. examples/runners.jl`).

- **Stopping.** `tol` stops once `loss(learner) <= tol`; `max_npoints` stops once the learner holds
  that many points, and `npoints` plus the evaluations in flight never exceeds it; `duration`
  (seconds or a `Dates.Period`) and `end_time` (a `DateTime`) stop at a deadline. A runner also
  stops when `ask` returns no points and nothing is in flight (the learner has reached its numerical
  resolution), so it cannot spin forever. `BlockingRunner` requires at least one goal; an
  `AsyncRunner` without one runs until [`cancel!`](@ref).
- **Refill per point.** The runner keeps `ntasks` evaluations in flight and asks for a new point
  as soon as one finishes, so a slow point never holds back the others (upstream behaviour).
- **Executors.** `executor` decides where `f` runs: any callable `executor(f, x)` returning a
  `fetch`-able handle. The default [`ThreadedExecutor`](@ref) spawns on the threads of this process
  with `ntasks = 1`; pass `ntasks = Threads.nthreads()` (and start Julia with `-t auto`) for a
  threaded run, with an `f` that is safe to call concurrently. The learner itself is never touched
  concurrently: only the runner loop calls `ask` and `tell!`.
- **Checkpoints.** `checkpoint = path` saves the data with [`save_data`](@ref) every
  `checkpoint_interval` seconds and when the runner stops.
- **Asynchronous execution.** The runner loop is an `@async` task pinned to the caller's thread;
  evaluations run in spawned tasks. So the learner changes only when the caller yields (`sleep`,
  `wait`, I/O), and reading `runner.learner` from the calling task needs no lock. Do not read it from
  another thread while the runner is running.
- **Progress.** [`status`](@ref) is `:running`, `:finished`, `:cancelled` or `:failed`; progress is
  `npoints(runner.learner)` and `loss(runner.learner)`.
- **Errors.** If `f` throws, results that already arrived are told; then the runner fails with an
  [`EvaluationError`](@ref) carrying the failing `point` and the original exception with its
  backtrace. `wait(runner)` (and `BlockingRunner`) rethrows it.
- **Stopping, any reason.** [`cancel!`](@ref), a deadline, a goal or a failure stop the runner
  immediately: evaluations still running are abandoned (they cannot be interrupted, and their
  results are dropped), and `remove_unfinished!(learner)` clears their pending points.

## Pending points and the shared interface

`LearnerND`, `AverageLearner`, `AverageLearner1D`, `IntegratorLearner` and `SequenceLearner`
implement the full interface below; `Learner1D` and `Learner2D` implement the core part (the
[`AbstractLearner`](@ref) docstring lists both).

- [`pending_points`](@ref)`(learner)` returns the points handed out by `ask` and not yet told.
- [`remove_unfinished!`](@ref)`(learner)` forgets them, so that `ask` can hand them out again. Use it
  after abandoned evaluations.
- `loss(learner; real = false)` counts pending points as if they were evaluated; `real = true`
  (default) uses told data only.
- `AdaptivePort.ask_with_improvements(learner, n)` returns upstream's pair `(points,
  loss_improvements)`; `ask` returns only the points.
- Some learners know when they are finished (the mean is accurate enough, the integral converged,
  every element is told). They define `done(learner)` without `tol` (`AdaptivePort.has_own_goal`
  is `true`), and a runner without `tol` stops on that; `BlockingRunner` then needs no other goal.


## Executors

An executor is any callable `executor(f, x)` that starts the evaluation and returns a handle `fetch` can
wait on (a `Task`, a `Distributed.Future`, a Dagger thunk, ...); `f` is the learner's function and `x` the
point (splatted for tuple points). Optionally define `AdaptivePort.default_ntasks(executor)`
([`default_ntasks`](@ref)), the number of evaluations to keep in flight. The runner needs nothing else, so a
custom executor is a few lines:

```julia
struct SpawnExecutor end                                  # runs every call in its own task
(::SpawnExecutor)(f, x) = Threads.@spawn f(x...)
AdaptivePort.default_ntasks(::SpawnExecutor) = Threads.nthreads()
BlockingRunner(learner; tol = 0.01, executor = SpawnExecutor())
```

The default is [`ThreadedExecutor`](@ref) (`examples/runners.jl`, `julia -t 4`). Evaluating on processes or
ranks needs a package; those executors live in package extensions that load with the package. Measured
speed-ups are in the [Performance](index.md#Performance) paragraph.

### Distributed.jl workers

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

### Dagger.jl

`DaggerExecutor` sends every evaluation to Dagger's scheduler (`Dagger.spawn(f, options, x...)`), which places it on
a thread or worker process it knows. Use it when you already use Dagger, or want threads and Distributed workers
behind one executor. Load the package first (`using Dagger`, package extension).

```julia
using Distributed, AdaptivePort
addprocs(2)
@everywhere using Dagger
using Dagger
Dagger.addprocs!(Dagger.Sch.eager_context(), workers())
learner = Learner1D(x -> tanh(50x); bounds = (-1, 1))
BlockingRunner(learner; tol = 1e-2, executor = DaggerExecutor())
```

- Keywords go to `Dagger.Options`, e.g. `DaggerExecutor(; scope = Dagger.ProcessScope(2))` restricts evaluation to
  worker 2. `default_ntasks` is `Dagger.num_processors(scope)`.
- Dagger does not notice workers added after its scheduler context was created; call
  `Dagger.addprocs!(Dagger.Sch.eager_context(), workers())` (or add workers before first use). Otherwise tasks fail
  with a `SchedulingException`.
- `f` must be defined on every worker that may run it (`@everywhere`).
- A throwing `f` yields an `EvaluationError` holding the original exception and its backtrace from inside `f`.
  There are no retries.

### MPI.jl (`MPIExecutor`)

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

Run it with `mpiexec -n 4 julia --project script.jl` (`examples/mpi.jl`); use the `mpiexec` that matches your MPI
library. With MPI.jl's bundled MPICH:

```julia
using MPI
MPI.mpiexec(m -> run(Cmd([m, "-n", "4", "julia", "--project=examples", "examples/mpi.jl"])))
```

`ntasks` defaults to the number of ranks minus one.

- `MPIExecutor(body, f; comm = MPI.COMM_WORLD)` returns the value of `body` on rank 0 and `nothing` on the others.
  Code after the block runs on every rank, so guard rank-0 work with `MPI.Comm_rank(comm) == 0`.
- Every rank holds `f` itself; only points and results are sent (serialized Julia objects). The executor throws an
  `ArgumentError` if called with another function, so pass the learner's own `learner.f`.
- A throwing `f` is reported to rank 0 as an `EvaluationError` and the worker keeps serving.
- Leaving the block, normally or by an exception, sends a stop message to every worker. Workers finish the
  evaluation in progress first, so leaving takes as long as the slowest running evaluation; its result is dropped.
- At least 2 ranks are needed (`ArgumentError` otherwise). A crashed rank is not detected: the run hangs.
  All MPI calls run on the thread that called `MPI.Init`; call the executor from that thread.


## Checkpoint and resume

`checkpoint = path` writes the evaluated data every `checkpoint_interval` seconds and when the runner stops
(see [Saving and export](saving.md)). To resume, build a fresh learner with the same bounds and call
[`load_data!`](@ref) before starting the runner; it continues where the data ends. A time goal (`duration`,
`end_time`) is a natural unit for one session of a long study. `examples/save_load.jl` runs two sessions
this way.

## Slurm job scripts

These scripts are shown, not run here; adjust partition, account and time to your cluster.

**Distributed workers inside one allocation.**
[SlurmClusterManager.jl](https://github.com/JuliaParallel/SlurmClusterManager.jl) starts one worker per Slurm
task; the driver script runs once:

```julia
# run.jl
using Distributed, SlurmClusterManager
addprocs(SlurmManager())                 # one worker per task of the allocation
@everywhere using AdaptivePort
@everywhere f(x) = expensive(x)          # f and its dependencies on every worker
learner = Learner1D(f; bounds = (-1, 1))
BlockingRunner(learner; tol = 1e-3, executor = DistributedExecutor(), checkpoint = "result.csv")
```

```bash
#!/bin/bash
#SBATCH --job-name=adaptive
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
julia --project run.jl                   # SlurmManager launches the workers itself
```

**MPI ranks.** With [`MPIExecutor`](@ref) the script of the MPI section above is the job; Slurm starts all
ranks:

```bash
#!/bin/bash
#SBATCH --job-name=adaptive-mpi
#SBATCH --ntasks=32
#SBATCH --time=02:00:00
srun julia --project mpi_script.jl       # or: mpiexec -n $SLURM_NTASKS julia --project mpi_script.jl
```

Rank 0 runs the learner and saves. Use `checkpoint` so that a job killed at the time limit loses little.
