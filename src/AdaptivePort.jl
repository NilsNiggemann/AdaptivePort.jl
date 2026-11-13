"""
    AdaptivePort

Julia port of Python Adaptive: adaptive sampling of functions through an `ask`/`tell!` interface. Provides
[`Learner1D`](@ref) (interval), [`Learner2D`](@ref) (rectangle) and [`LearnerND`](@ref) (box), all with scalar or
vector values; [`AverageLearner`](@ref) and [`AverageLearner1D`](@ref) for noisy functions; [`IntegratorLearner`](@ref)
(adaptive quadrature); [`SequenceLearner`](@ref) (a fixed list of inputs); and runners ([`BlockingRunner`](@ref),
[`AsyncRunner`](@ref)) that evaluate the function for you.
"""
module AdaptivePort

import Dates
import DelaunayTriangulation as DT
import LinearAlgebra
import Random
import Serialization
import StatsFuns

export AbstractLearner, ask, tell!, tell_pending!, loss, done, result, npoints, pending_points, remove_unfinished!
export Learner1D, Learner2D, default_loss, triangles
export LearnerND, default_loss_nd, uniform_loss_nd, std_loss_nd, simplices
export AverageLearner, AverageLearner1D, nsamples, IntegratorLearner, DivergentIntegralError, SequenceLearner
export Runner, AsyncRunner, BlockingRunner, EvaluationError, status, cancel!
export ThreadedExecutor, DistributedExecutor, DaggerExecutor, MPIExecutor, default_ntasks
export BalancingLearner
export to_table, save_data, load_data!
export DataSaver, extra_data

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

- `pending_points(learner)`: `Vector` of the points issued but not yet told.
- `remove_unfinished!(learner)`: forget all pending points; returns `learner`. Runners call it whenever they stop.
- `loss(learner; real = true)`: `real = false` also counts the pending points as if they were evaluated.
- `AdaptivePort.ask_with_improvements(learner, n; tell_pending = true)`: `(points, loss_improvements)`, upstream's
  return value of `ask`. Defining it gives `ask` for free.
- `to_table(learner)`: the evaluated data as a column table (named tuple of equal-length vectors), one row per `tell!`.
- `AdaptivePort.table_rows(learner, table)`: the `(point, value)` pairs that replay `table` through `tell!`.

**Interface (optional)**
- `done(learner; tol)`: defaults to `loss(learner) <= tol`.
- `AdaptivePort.has_own_goal(learner)`: `true` if `done(learner)` without `tol` is meaningful (the learner knows when it
  is finished); default `false`. Runners then stop on `done(learner)` when no `tol` is given.
- `AdaptivePort.load_table!(learner, table)`: tell every row of `table` (a [`to_table`](@ref) result); the default
  replays `table_rows`. Override when replaying in order is not enough (`IntegratorLearner`).
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

"""
    ask(learner, n; tell_pending = true)

`Vector` of up to `n` new points to evaluate, never ones already evaluated or pending (fewer, or none, once the
learner has reached its resolution). With `tell_pending = true` the points are marked pending, so further calls do
not return them again before they are told; `false` only peeks. Returns the points only; use
[`AdaptivePort.ask_with_improvements`](@ref) for upstream's pair with the loss improvements.

**See Also**
- [`tell!`](@ref), [`pending_points`](@ref)
"""
ask(l::AbstractLearner, n::Integer; tell_pending = true) = first(ask_with_improvements(l, n; tell_pending))

"""
    to_table(learner)

The evaluated data of `learner` as a column table: a named tuple of equal-length `Vector`s, one row per told
evaluation (upstream `to_dataframe`). It is a Tables.jl column table, so `DataFrame(to_table(learner))` and
`CSV.write(path, to_table(learner))` work without this package depending on either. Never mutates `learner`.

Vector-valued output gives one numbered column per component.

**See Also**
- [`save_data`](@ref), [`load_data!`](@ref)
"""
function to_table end

"""
    table_rows(learner, table)

Iterator of `(point, value)` pairs such that `tell!(learner, point, value)` for each pair restores the data of
`table` (a [`to_table`](@ref) result of a learner of the same kind). Unexported; part of the learner interface.
"""
function table_rows end

"""
    load_table!(learner, table)

Tell `learner` every row of `table`, a [`to_table`](@ref) result of a learner of the same kind. The default replays
[`table_rows`](@ref AdaptivePort.table_rows) through `tell!`. Unexported. Returns `learner`.
"""
function load_table!(l::AbstractLearner, table)
    for (x, y) in table_rows(l, table)
        tell!(l, x, y)
    end
    return l
end

"""
    default_ntasks(executor)

Number of evaluations a runner keeps in flight with `executor` when `ntasks` is not given: `1` unless the executor
knows its worker count (upstream uses the executor's core count).
"""
default_ntasks(executor) = 1

include("executors.jl")
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
include("data_saver.jl")
include("balancing_learner.jl")
include("io.jl")
include("runner.jl")

end
