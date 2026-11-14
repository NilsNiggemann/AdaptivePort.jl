# AdaptivePort.jl

AdaptivePort.jl samples an expensive function where it is most informative, instead of on a fixed grid. It is
an independent Julia implementation of [Python Adaptive](https://github.com/python-adaptive/adaptive).

![Learner1D versus a uniform grid](assets/learner1d_vs_uniform.png)

*Three sharp functions, each sampled with 50 evaluations. Top: uniform grid. Bottom: `Learner1D` puts the
points where the function changes.*

!!! warning "Early-stage package"
    The package is experimental and much of its code is AI-generated. It is tested against the behaviour of
    Python Adaptive, but it is not registered yet and its API may still change.

## How it works

Two objects share the work:

- A **learner** decides *where* to sample next. It knows the domain and the values seen so far, and never
  calls the function itself.
- A **runner** decides *when and how* to evaluate: it repeatedly asks the learner for points, evaluates the
  function (in this process, on threads, on Distributed workers or on MPI ranks) and tells the learner the
  results.

You can also drive a learner by hand with `ask` and `tell!`, which is often the simplest way to plug it into
existing code. [Getting started](getting_started.md) shows both.

## Installation

The package is not registered. Install it from the git URL:

```julia
using Pkg
Pkg.add(url = "https://github.com/NilsNiggemann/AdaptivePort.jl")
```

## Quick start

```julia
using AdaptivePort

f(x) = tanh(50x) + 0.2sin(8x)
learner = Learner1D(f; bounds = (-1, 1))
BlockingRunner(learner; tol = 0.02, max_npoints = 500)   # ask, evaluate, tell until done
data = result(learner)                                    # (x = sorted points, y = values)
```

## Choosing a learner

| You want ... | Use | Points are |
|:-------------|:----|:-----------|
| a function of one variable on an interval | [`Learner1D`](@ref) | `x` |
| a function of two variables on a rectangle | [`Learner2D`](@ref) | `(x, y)` |
| a function on a box in $N \le 4$ dimensions | [`LearnerND`](@ref) | `NTuple{N}` |
| the mean of a noisy quantity | [`AverageLearner`](@ref) | integer seed |
| a noisy function of one variable | [`AverageLearner1D`](@ref) | `(seed, x)` |
| the integral of a function on an interval | [`IntegratorLearner`](@ref) | `x` |
| a parallel `map` over a fixed list of inputs | [`SequenceLearner`](@ref) | `(index, element)` |
| several learners sharing one runner and budget | [`BalancingLearner`](@ref) | `(index, point)` |
| to keep extra output (errors, diagnostics) of `f` | [`DataSaver`](@ref) around any learner | as the wrapped learner |

!!! tip "Vector-valued functions"
    `Learner1D`, `Learner2D` and `LearnerND` also learn functions with several outputs (several order parameters,
    real and imaginary part, ...): pass `vdim = M`. See [Vector-valued functions](learners/learner1d.md#Vector-valued-functions).

## How these docs are organised

- **Introduction**: this page and the [Getting started](getting_started.md) tutorial.
- **Examples**: [Plotting results](examples/plotting.md) and a [phase diagram on 4 cluster
  nodes](examples/phase_diagram.md).
- **Learners**: one page per learner, with an example, the algorithm and its limits, plus the
  [common interface](learners/interface.md) all learners share.
- **Runners**: [running a learner](runners/runners.md), [parallel execution](runners/parallel.md) on threads,
  processes and MPI ranks, and [clusters](runners/clusters.md) (Slurm).
- **Saving data**: [saving, loading and export](saving.md) to tables and CSV.
- **Reference**: [compatibility](compatibility.md) with Python Adaptive and the [API reference](api.md).

Runnable scripts for every page are in
[`examples/`](https://github.com/NilsNiggemann/AdaptivePort.jl/tree/main/examples). Most run with
`julia --project=. examples/<name>.jl`. The plotting, Distributed, Dagger, MPI and phase-diagram scripts need the
examples environment: `julia --project=examples ...`.

## Performance

`Learner1D` costs about 90 µs per point up to $10^4$ points. `Learner2D`, asked one point at a time, averages about
5 ms per point over a run of 1000 points and 15 ms over 3000, mostly for one Delaunay triangulation per `ask`. `BlockingRunner` adds about 10 µs per point
compared with a hand-written loop. A function that takes 10 ms runs 4.0 times faster on 4 threads and 1.85 times
faster on 2 Distributed workers. Saving $10^6$ points takes about 1 s as CSV and 0.09 s as `.jls`. Details and the
benchmark scripts are in
[`benchmark/README.md`](https://github.com/NilsNiggemann/AdaptivePort.jl/blob/main/benchmark/README.md).
Parallel runs pay off only for expensive functions; see [Parallel execution](runners/parallel.md).

## Relation to Python Adaptive

The package follows upstream's algorithms and API conventions. Upstream's behaviour is the reference, but its
architecture is not copied: learners never call the function, `ask` returns points only, and runners are plain
Julia tasks with callable executors. [Compatibility](compatibility.md) lists what is the same, what differs on
purpose and what is missing. Python Adaptive is distributed under the BSD 3-Clause License (Copyright (c)
2017-2023, Adaptive authors).
