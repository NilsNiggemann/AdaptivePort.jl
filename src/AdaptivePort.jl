"""
    AdaptivePort

Julia port of Python Adaptive: adaptive sampling of scalar functions through an `ask`/`tell!` interface. Provides [`Learner1D`](@ref) (interval) and [`Learner2D`](@ref) (rectangle).
"""
module AdaptivePort

import DelaunayTriangulation as DT
import Random

export AbstractLearner, Learner1D, Learner2D, ask, tell!, tell_pending!, loss, done, result, npoints, default_loss, triangles

"""
    AbstractLearner

Adaptive sampler of a function the caller evaluates: the learner proposes points, the caller evaluates them and reports back.

**Interface**
- `ask(learner, n; tell_pending = true)`: `Vector` of up to `n` new points, never ones already evaluated or pending.
- `tell!(learner, x, y)`: record the evaluation `y = f(x)`; returns `learner`.
- `tell_pending!(learner, x)`: mark `x` as issued but not evaluated; returns `learner`.
- `loss(learner)`: largest remaining loss, `Inf` until the learner covers its domain.
- `npoints(learner)`: number of evaluated points.
- `result(learner)`: copies of the evaluated data as a named tuple.

**Interface (optional)**
- `done(learner; tol)`: defaults to `loss(learner) <= tol`.
"""
abstract type AbstractLearner end

"""
    done(learner; tol)

Whether `loss(learner) <= tol`.

**Keyword Arguments**
- `tol`: loss tolerance; required because the loss scale depends on the loss function.

**See Also**
- [`loss`](@ref)
"""
done(l::AbstractLearner; tol) = loss(l) <= tol

include("learner1D.jl")
include("triangulation.jl")
include("learner2D.jl")

end
