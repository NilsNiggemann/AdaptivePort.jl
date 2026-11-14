# SequenceLearner

[`SequenceLearner`](@ref) evaluates a function on a fixed list of inputs. It is a plain `map` that
runs through the runners: parallel, cancellable, with the same interface as the adaptive learners.
The points are `(index, element)` with a 1-based `index`.

```julia
using AdaptivePort

learner = SequenceLearner(length; sequence=["a", "bb", "ccc"], value_type=Int)
BlockingRunner(learner; ntasks=2)

data = result(learner)     # (indices, values) of the told elements
data.values                # [1, 2, 3]
```

`examples/sequence.jl` is the runnable script.

!!! tip "Use it for parallel maps with checkpoints"
    `SequenceLearner` is the simplest way to get the runner's features for a plain list of jobs: parallel
    executors, time limits, cancellation and [checkpoints](../saving.md). Pass a concrete `value_type` for a
    concretely typed result.

## Advanced: algorithm

There is nothing adaptive: `ask` returns the lowest indices that are neither told nor pending. The
loss is the fraction of elements not told yet, so `done(learner)` (without `tol`) means that every
element is told, and a runner without `tol` stops then. `result` works at any time and returns the
told elements only, in index order. `value_type` (default `Any`) sets the element type of the
stored values; use a concrete type for a concretely typed result.

## Limits

Telling an index again is ignored. Differences to upstream:
[Deviations](../compatibility.md#Deviations-from-Python-Adaptive).

## API

```@docs
SequenceLearner
```
