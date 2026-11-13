# AdaptivePort.jl

Sample an expensive function where it is most informative instead of on a fixed grid. AdaptivePort.jl is an
independent Julia implementation of [Python Adaptive](https://github.com/python-adaptive/adaptive), an early-stage
experimental package with much AI-generated code. A *learner* decides where to sample next; you (or a
[runner](runners.md)) evaluate the function and tell the learner the value.

## Installation

The package is not registered; install it from the git URL:

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

The same loop written by hand, which is always supported, is in [Learner1D](learner1d.md). Runnable scripts are in
[`examples/`](https://github.com/NilsNiggemann/AdaptivePort.jl/tree/main/examples): `learner1d.jl`,
`learner2d.jl`, `integrator.jl`, `balancing.jl`, `datasaver.jl`, `runners.jl` (blocking, asynchronous and
threaded), `distributed.jl`, `dagger.jl`, `mpi.jl`, `save_load.jl`, and the plotting scripts
(`learner1d_plot.jl`, `learner2d_rings.jl`, `learner2d_spinice.jl`, `learnernd_shell.jl`; run with
`julia --project=examples`).

## Learners

| Learner | Samples | Page |
|:--------|:--------|:-----|
| [`Learner1D`](@ref) | a function on an interval | [Learner1D](learner1d.md) |
| [`Learner2D`](@ref) | a function on a rectangle | [Learner2D](learner2d.md) |
| [`LearnerND`](@ref) | a function on a box in $N$ dimensions | [LearnerND](learnernd.md) |
| [`AverageLearner`](@ref) | the mean of a noisy number | [AverageLearner](average_learner.md) |
| [`AverageLearner1D`](@ref) | a noisy function on an interval | [AverageLearner1D](average_learner1d.md) |
| [`IntegratorLearner`](@ref) | the integral of a function on an interval | [IntegratorLearner](integrator.md) |
| [`SequenceLearner`](@ref) | a plain map over a fixed list of inputs | [SequenceLearner](sequence.md) |
| [`BalancingLearner`](@ref) | several learners served by one runner | [Balancing](balancing.md) |
| [`DataSaver`](@ref) | wraps a learner for a function with extra output | [DataSaver](datasaver.md) |

All implement the [`AbstractLearner`](@ref) interface. Other pages: [Runners and parallel execution](runners.md),
[Saving and export](saving.md), [Plotting](plotting.md), [Compatibility](compatibility.md) with Python Adaptive
(including the intentional deviations) and the [API reference](api.md).

![Learner1D versus a uniform grid](assets/learner1d_vs_uniform.png)

## Performance

Learner1D runs at about 90 us per point up to 1e4 points. Learner2D costs about 5 ms per point at 1000 points
and 15 ms at 3000, dominated by one Delaunay triangulation per ask. BlockingRunner adds about 10 us per point
over a hand-written loop. A 10 ms function speeds up 4.0x with 4 threads and 1.85x with 2 Distributed workers.
Saving 1e6 points takes about 1 s as CSV and 0.09 s as .jls. Details and the benchmark scripts:
[`benchmark/README.md`](https://github.com/NilsNiggemann/AdaptivePort.jl/blob/main/benchmark/README.md).

## Relation to Python Adaptive

The package follows upstream's algorithms and API conventions; upstream behaviour is the reference, its
architecture is not. Learners never call the function themselves, `ask` returns points only, and runners are
plain Julia tasks with callable executors. [Compatibility](compatibility.md) lists what is the same, what
differs on purpose, and what is missing. Python Adaptive is distributed under the BSD 3-Clause License
(Copyright (c) 2017-2023, Adaptive authors).
