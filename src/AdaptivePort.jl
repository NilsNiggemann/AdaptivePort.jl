"""
    AdaptivePort

Julia port of Python Adaptive: adaptive sampling of functions through an `ask`/`tell!` interface. Provides
[`Learner1D`](@ref) (interval), [`Learner2D`](@ref) (rectangle) and [`LearnerND`](@ref) (box), all with scalar or
vector values; [`AverageLearner`](@ref) and [`AverageLearner1D`](@ref) for noisy functions; [`IntegratorLearner`](@ref)
(adaptive quadrature); [`SequenceLearner`](@ref) (a fixed list of inputs); and runners ([`BlockingRunner`](@ref),
[`AsyncRunner`](@ref)) that evaluate the function for you.
"""
module AdaptivePort

import DelaunayTriangulation as DT
import LinearAlgebra
import Random
import StatsFuns

export AbstractLearner, ask, tell!, tell_pending!, loss, done, result, npoints, pending_points, remove_unfinished!
export Learner1D, Learner2D, default_loss, triangles
export LearnerND, default_loss_nd, uniform_loss_nd, std_loss_nd, simplices
export AverageLearner, AverageLearner1D, nsamples, IntegratorLearner, SequenceLearner
export Runner, AsyncRunner, BlockingRunner, EvaluationError, status, cancel!

"""
    AbstractLearner

Adaptive sampler of a function the caller evaluates: the learner proposes points, the caller evaluates them and reports back.

**Interface**
- `ask(learner, n; tell_pending = true)`: `Vector` of up to `n` new points, never ones already evaluated or pending.
- `tell!(learner, x, y)`: record the evaluation `y = f(x...)`; returns `learner`.
- `tell_pending!(learner, x)`: mark `x` as issued but not evaluated; returns `learner`.
- `loss(learner)`: largest remaining loss, `Inf` until the learner covers its domain.
- `npoints(learner)`: number of evaluated points.
- `result(learner)`: copies of the evaluated data as a named tuple.
- field `f`: the function, called by runners as `f(x...)` on a point `x` from `ask`.

**Interface (Stage 5; implemented by `LearnerND`, `AverageLearner`, `AverageLearner1D`, `IntegratorLearner`, `SequenceLearner`)**
- `pending_points(learner)`: `Vector` of the points issued but not yet told.
- `remove_unfinished!(learner)`: forget all pending points; returns `learner`.
- `loss(learner; real = true)`: `real = false` also counts the pending points as if they were evaluated.
- `AdaptivePort.ask_with_improvements(learner, n; tell_pending = true)`: `(points, loss_improvements)`, upstream's
  return value of `ask`. Defining it gives `ask` for free.

**Interface (optional)**
- `done(learner; tol)`: defaults to `loss(learner) <= tol`.
- `AdaptivePort.has_own_goal(learner)`: `true` if `done(learner)` without `tol` is meaningful (the learner knows when it
  is finished); default `false`. Runners then stop on `done(learner)` when no `tol` is given.
"""
abstract type AbstractLearner end

"""
    done(learner; tol)

Whether `loss(learner) <= tol`. Learners with their own stopping criterion ([`AverageLearner`](@ref),
[`IntegratorLearner`](@ref), [`SequenceLearner`](@ref)) also accept `done(learner)` without `tol`.

**Keyword Arguments**
- `tol`: loss tolerance; required for other learners because the loss scale depends on the loss function.

**See Also**
- [`loss`](@ref)
"""
done(l::AbstractLearner; tol) = loss(l) <= tol

"""
    has_own_goal(learner)

Whether `learner` defines `done(learner)` without a tolerance; `false` unless a learner opts in. Unexported.
"""
has_own_goal(::AbstractLearner) = false

"""
    pending_points(learner)

Points handed out by `ask` (or registered with [`tell_pending!`](@ref)) and not yet told, as a new `Vector`.

**See Also**
- [`remove_unfinished!`](@ref)
"""
function pending_points end

"""
    remove_unfinished!(learner)

Forget all pending points, so that `ask` may hand them out again (upstream `remove_unfinished`). Use it after
evaluations were abandoned. Returns `learner`.

**See Also**
- [`pending_points`](@ref)
"""
function remove_unfinished! end

"""
    ask_with_improvements(learner, n; tell_pending = true)

Upstream's `ask`: up to `n` new points together with the expected loss improvement of each, as
`(points::Vector, improvements::Vector)`. The public [`ask`](@ref) returns only the points. Unexported.
"""
function ask_with_improvements end

ask(l::AbstractLearner, n::Integer; tell_pending = true) = first(ask_with_improvements(l, n; tell_pending))

include("learner1D.jl")
include("triangulation.jl")
include("learner2D.jl")
include("triangulationND.jl")
include("learnerND.jl")
include("average_learner.jl")
include("average_learner1D.jl")
include("integrator_coeffs.jl")
include("integrator_learner.jl")
include("sequence_learner.jl")
include("runner.jl")

end
