# Getting started

This tutorial samples a function with a sharp step, first with a runner and then by hand. It then runs the
function in parallel, on threads and on worker processes, and saves the data. Every step builds on the previous
one. The full scripts are `examples/learner1d.jl`, `examples/runners.jl` and `examples/distributed.jl`.

## Step 1: Define the function and a learner

```julia
using AdaptivePort

f(x) = tanh(50x) + 0.2sin(8x)                # steep near x = 0, smooth elsewhere
learner = Learner1D(f; bounds = (-1, 1))
```

The learner stores `f` for the runners, but never calls it itself. `bounds` is the interval to sample.

## Step 2: Run it

```julia
BlockingRunner(learner; tol = 0.02, max_npoints = 500)
```

`BlockingRunner` asks the learner for a point, evaluates `f` there, tells the learner the value, and repeats. It
returns when one of its goals is met:

- `tol`: the learner's [`loss`](@ref) is at most `tol`. The loss measures how much the sampling could still
  improve; for `Learner1D` it is the longest segment of the sampled curve in coordinates scaled to the unit
  square.
- `max_npoints`: the learner holds that many points.
- `duration` or `end_time`: a time limit.

!!! warning "Always set `max_npoints` or a time limit"
    A `tol` that is too small for the function (for example on a discontinuity) keeps refining until the
    numerical resolution runs out. A point or time limit bounds the cost of a run.

## Step 3: Look at the result

```julia
data = result(learner)     # (x = sorted points, y = values)
npoints(learner)           # number of evaluated points
loss(learner)              # current loss
```

The points concentrate around $x = 0$, where `f` is steep. [Plotting results](examples/plotting.md) shows how to
draw them.

## Step 4: The same loop by hand

A runner only automates the `ask` → evaluate → `tell!` loop. Writing it yourself is always supported and is the
easiest way to call code that is not a plain Julia function (an external program, a lab instrument, a cluster
queue):

```julia
learner = Learner1D(f; bounds = (-1, 1))
while !done(learner; tol = 0.02) && npoints(learner) < 500
    points = ask(learner, 4)            # up to 4 new points
    isempty(points) && break            # numerical resolution reached
    for x in points
        tell!(learner, x, f(x))         # evaluate any way you like
    end
end
```

`done(learner; tol)` is `loss(learner) <= tol`. `ask` marks the points it returns as *pending*: handed out, but
not yet *told* (reported with `tell!`). The next `ask` therefore returns different points, even before the first
ones are told. That is what makes batches and parallel evaluation possible. All learners share these functions;
see the [common interface](learners/interface.md).

!!! note "Points of other learners"
    `Learner2D` and `LearnerND` points are tuples: `for p in ask(learner, n); tell!(learner, p, f(p...)); end`.
    The table on the [home page](index.md#Choosing-a-learner) lists the point type of every learner.

## Step 5: Run on threads

Start Julia with several threads (`julia -t 4`) and let the runner keep 4 evaluations in flight:

```julia
learner = Learner1D(f; bounds = (-1, 1))
BlockingRunner(learner; tol = 0.02, max_npoints = 500, ntasks = Threads.nthreads())
```

!!! warning "Thread safety"
    With `ntasks > 1` the function runs on several threads at once, so it must be safe to call concurrently (no
    shared mutable buffers without locks). The learner itself is never touched concurrently.

## Step 6: Run on worker processes or cluster nodes

Worker processes avoid thread-safety issues and scale to many machines. Pass an `executor`:

```julia
using Distributed, AdaptivePort
addprocs(4)                              # or one worker per Slurm task on a cluster
@everywhere using AdaptivePort
@everywhere f(x) = tanh(50x) + 0.2sin(8x)   # f must be defined on every worker
learner = Learner1D(f; bounds = (-1, 1))
BlockingRunner(learner; tol = 0.02, max_npoints = 500, executor = DistributedExecutor())
```

On a cluster, the same script runs with workers on all nodes of a Slurm allocation; see
[Clusters](runners/clusters.md). MPI and Dagger work through the `executor` keyword too; see
[Parallel execution](runners/parallel.md).

## Step 7: Save and resume

```julia
save_data("run.csv", learner)                 # told points as CSV

fresh = Learner1D(f; bounds = (-1, 1))
load_data!(fresh, "run.csv")                  # replays the data
BlockingRunner(fresh; tol = 0.01, max_npoints = 1000)   # continues where the data ends
```

A runner can also save on its own every few minutes with `checkpoint = "run.csv"`. See
[Saving data](saving.md).

## Where to go next

- [Scan a phase diagram on a cluster](examples/phase_diagram.md): a two-dimensional scan with several order
  parameters on 4 Slurm nodes.
- The page of [your learner](index.md#Choosing-a-learner): its options, algorithm and limits.
- [Running a learner](runners/runners.md): goals, asynchronous runs, progress, errors.
