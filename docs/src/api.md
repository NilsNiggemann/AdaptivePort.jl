# API reference

Docstrings of all exported and selected internal names. The docstrings of [`LearnerND`](@ref),
[`AverageLearner`](@ref), [`AverageLearner1D`](@ref), [`IntegratorLearner`](@ref) and [`SequenceLearner`](@ref)
are at the end of their [learner pages](learners/learnernd.md); everything else is here.

## Learner interface

```@docs
AdaptivePort
AbstractLearner
ask
tell!
tell_pending!
loss
done
result
npoints
pending_points
remove_unfinished!
```

## Learners

```@docs
Learner1D
Learner2D
default_loss
triangles
BalancingLearner
AdaptivePort.Dispatch
AdaptivePort.from_product
DataSaver
extra_data
```

## Runners

```@docs
Runner
BlockingRunner
AsyncRunner
EvaluationError
status
cancel!
Base.wait(::Runner)
```

## Executors

```@docs
ThreadedExecutor
DistributedExecutor
DaggerExecutor
MPIExecutor
default_ntasks
```

## Saving and export

```@docs
to_table
save_data
load_data!
```

## Internals

Unexported functions of the learner interface and implementation helpers.

```@docs
AdaptivePort.ask_with_improvements
AdaptivePort.has_own_goal
AdaptivePort.table_rows
AdaptivePort.load_table!
AdaptivePort.captured_exception
AdaptivePort.deviations
AdaptivePort.estimate_gradients
AdaptivePort.choose_point_in_triangle
AdaptivePort.delaunay
AdaptivePort.delaunay_triangles
AdaptivePort.signed_area
AdaptivePort.interpolate_linear
AdaptivePort.update_value!
```
