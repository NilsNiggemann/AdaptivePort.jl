# Common interface

All learners are subtypes of [`AbstractLearner`](@ref) and share one interface, so runners, saving and
[`BalancingLearner`](@ref) work with any of them.

## Basic functions

| Function | Does |
|:---------|:-----|
| [`ask`](@ref)`(learner, n)` | returns up to `n` new points and marks them pending |
| [`tell!`](@ref)`(learner, x, y)` | records the value `y` at the point `x` |
| [`loss`](@ref)`(learner)` | how much the sampling could still improve; smaller is better |
| [`done`](@ref)`(learner; tol)` | `loss(learner) <= tol` |
| [`npoints`](@ref)`(learner)` | number of evaluated points |
| [`result`](@ref)`(learner)` | the data, as a named tuple of copies |

The learner never calls the function. The caller evaluates it, or a runner calls `learner.f(x...)`.

## Pending points

`ask` hands out points and remembers them as *pending* until they are told. This is what lets several
evaluations run at once without duplicates.

- [`pending_points`](@ref)`(learner)` returns the points handed out by `ask` and not yet told.
- [`tell_pending!`](@ref)`(learner, x)` marks a point as pending without `ask`, for points you chose yourself.
- [`remove_unfinished!`](@ref)`(learner)` forgets the pending points, so `ask` can hand them out again. Call it
  after abandoned evaluations; runners do this when they stop.
- `loss(learner; real = false)` counts pending points as if they were evaluated; `real = true` (default) uses
  told data only.

!!! note "Pending points are not saved"
    [`save_data`](@ref) writes told points only. After a crash, the pending points are simply asked again.

## Learners with their own goal

[`AverageLearner`](@ref), [`IntegratorLearner`](@ref) and [`SequenceLearner`](@ref) know when they are finished
(the mean is accurate enough, the integral converged, every element is told). For them `done(learner)` works
without `tol`, and a runner without `tol` stops when they are done. A `DataSaver` follows its child, and a
`BalancingLearner` whose children all have an own goal is done when all of them are. `AdaptivePort.has_own_goal(learner)` tells
which learners do.

## Advanced: implementing a learner

A new learner subtypes [`AbstractLearner`](@ref), stores its function in a field `f`, and implements the methods
in the **Interface** list of the [`AbstractLearner`](@ref) docstring: `ask` (or, more conveniently,
`AdaptivePort.ask_with_improvements`, which gives `ask` for free), `tell!`, `tell_pending!`, `loss`, `npoints`,
`result`, `pending_points`, `remove_unfinished!`, `to_table` and `AdaptivePort.table_rows`. Runners, saving and
`BalancingLearner` then work unchanged. All learners of this package implement the full interface.

`AdaptivePort.ask_with_improvements(learner, n)` returns upstream's pair `(points, loss_improvements)`; `ask`
returns only the points.
