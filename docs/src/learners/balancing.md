# BalancingLearner

[`BalancingLearner`](@ref) serves several learners of the same kind with one runner and always asks the child
whose next point promises the most (`strategy = :loss_improvements`, or `:loss`, `:npoints`, `:cycle`). Use it to
sample a family of functions (different parameters, different signals) under a shared budget. Points are
`(index, point)` with a 1-based child `index`, and `balancer.f` is a [`AdaptivePort.Dispatch`](@ref) that routes a
point to its child's function.

```julia
using AdaptivePort

fs = [x -> x, x -> sin(3x), x -> tanh(50x), x -> sin(40x) * exp(-x^2)]
balancer = BalancingLearner([Learner1D(f; bounds = (-1, 1)) for f in fs])
BlockingRunner(balancer; max_npoints = 400, ntasks = 4)
npoints.(balancer.learners)    # the hard functions got most of the points
```

This is `examples/balancing.jl`. [`AdaptivePort.from_product`](@ref) builds the children from a parameter grid:

```julia
b = AdaptivePort.from_product((x; a, b) -> a * x + b, Learner1D, (bounds = (-1, 1),), (a = 1:3, b = [0.0, 1.0]))
```

!!! warning "Tell children through the balancer"
    The balancer caches the children's proposals and losses. Telling a child directly leaves those caches
    stale; use `tell!(balancer, (i, x), y)`.

!!! tip "Choosing a strategy"
    All children must be the same kind of learner (for example all `Learner1D`); otherwise the constructor
    throws an `ArgumentError`. `:loss_improvements` (default) and `:loss` compare the children's losses. If the
    losses are not comparable (different `loss_per_interval`, say), use `:npoints` (equal point counts) or
    `:cycle` (round robin).

- `result(balancer)` is the vector of child results; `to_table` stacks the child tables with a `learner_index`
  column, and [`save_data`](@ref) writes the whole balancer to one file ([Saving and export](../saving.md)).
- `done(balancer)` without `tol` works when the children have their own goal (for example
  [`SequenceLearner`](@ref)); then the runner stops when all are done.
