# Running a learner

A runner does the `ask` → evaluate → `tell!` loop for you, using the function stored in the learner (`f(x)` for
[`Learner1D`](@ref), `f(x, y)` for [`Learner2D`](@ref), `f(point...)` in general). Construct a learner, construct a
runner, then read the result. The hand-written loop of [Getting started](../getting_started.md#Step-4:-The-same-loop-by-hand)
stays fully supported.

```julia
using AdaptivePort

f(x) = tanh(50x)
learner = Learner1D(f; bounds = (-1, 1))
BlockingRunner(learner; tol = 0.02, max_npoints = 500)   # returns when a goal is met
data = result(learner)
```

`examples/runners.jl` shows blocking, asynchronous and threaded runs (`julia -t 4 --project=. examples/runners.jl`).

## Goals: when a runner stops

| Keyword | Stops when |
|:--------|:-----------|
| `tol` | `loss(learner) <= tol` |
| `max_npoints` | the learner holds that many points; points in flight are counted, so it is never exceeded |
| `duration` | that many seconds (or a `Dates.Period`) have passed |
| `end_time` | that `DateTime` is reached |
| (none) | the learner is done by its own goal (`AverageLearner`, `IntegratorLearner`, `SequenceLearner`, a `DataSaver` around one, a `BalancingLearner` of them) |

A runner also stops when `ask` returns no points and nothing is in flight (the learner has reached its numerical
resolution), so it cannot spin forever.

At a stop, evaluations still in flight are abandoned and their results dropped. With `drain = t`, a stop at a goal
or the deadline (not `cancel!` or a failure) first waits up to `t` seconds for them and keeps their results; no new
points are started meanwhile. Points of slow functions are then not wasted at the end of a timed run.

!!! warning "`BlockingRunner` needs a goal"
    `BlockingRunner` throws an `ArgumentError` without any goal, unless the learner has its own. An
    `AsyncRunner` without a goal runs until [`cancel!`](@ref).

## Asynchronous runs

`AsyncRunner` returns at once and runs in the background of the REPL. Watch it, stop it or wait for it:

```julia
runner = AsyncRunner(learner; tol = 0.001)
status(runner)                              # :running, :finished, :cancelled or :failed
npoints(runner.learner), loss(runner.learner)   # progress
cancel!(runner)                             # stop now
wait(runner)                                # block until it stops; rethrows its error
```

In the REPL you can read `runner.learner` (plot it, save it) while the runner works, without a lock. This is
because the runner loop is an `@async` task on the caller's thread: the learner changes only while the caller
yields (`sleep`, `wait`, I/O). Evaluations run in separately spawned tasks.

!!! danger "Do not read the learner from another thread"
    While a runner is running, read the learner only from the task that started it (for example the REPL), never
    from `Threads.@spawn` code: the runner may be changing it at the same time.

## Evaluation in parallel

The runner keeps `ntasks` evaluations in flight and asks for a new point as soon as one finishes, so a slow point
never holds back the others. Where `f` runs is decided by the `executor`:

```julia
BlockingRunner(learner; tol = 0.01, ntasks = Threads.nthreads())   # threads (start Julia with -t auto)
BlockingRunner(learner; tol = 0.01, executor = DistributedExecutor())   # worker processes
```

The default [`ThreadedExecutor`](@ref) uses the threads of this process with `ntasks = 1`. See
[Parallel execution](parallel.md) for threads, Distributed.jl, Dagger and MPI.

## Errors

If `f` throws, the results that already arrived are told, and the runner fails with an [`EvaluationError`](@ref)
carrying the failing `point` and the original exception with its backtrace. `BlockingRunner` and `wait(runner)`
rethrow it.

!!! note "No retries"
    The first failing evaluation stops the run. To skip bad points instead, catch the error inside `f` and return
    `NaN`. This works for `Learner1D` (the interval is then ranked by width) and `Learner2D` (non-finite values are
    left out of the triangulation, but `NaN` at a corner leaves `ask` with nothing to refine, and the run ends).
    `LearnerND` rejects non-finite values with an `ArgumentError`.

## What happens when a runner stops

Whatever the reason (goal, deadline, [`cancel!`](@ref), error), the runner stops at once:

1. results that have already arrived are told;
2. evaluations still running are abandoned (they cannot be interrupted, and their results are dropped);
3. `remove_unfinished!(learner)` clears their pending points, so the next run asks them again;
4. if `checkpoint` is set, the data are saved a last time.

## Checkpoints and resuming

`checkpoint = path` saves the data with [`save_data`](@ref) every `checkpoint_interval` seconds and when the runner
stops. To resume, build a fresh learner with the same function and bounds, call [`load_data!`](@ref), and start a
runner again; it continues where the data end:

```julia
path = "run.csv"
learner = Learner1D(f; bounds = (-1, 1))
isfile(path) && load_data!(learner, path)
BlockingRunner(learner; tol = 1e-3, duration = 3600, checkpoint = path, checkpoint_interval = 60)
```

A time goal (`duration`, `end_time`) is a natural unit for one session of a long study, for example one cluster
job. `examples/save_load.jl` runs two sessions this way.
