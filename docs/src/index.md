# AdaptivePort.jl

Julia port of [Python Adaptive](https://github.com/python-adaptive/adaptive): sample a function
where it is most informative instead of on a fixed grid. Currently provides one learner,
[`Learner1D`](@ref), for scalar functions on an interval.

## Workflow

The learner decides *where* to sample; you decide *how* to evaluate the function.

```julia
using AdaptivePort

f(x) = sin(8x) + 0.2cos(23x)
learner = Learner1D(f; bounds=(-1, 1))

while !done(learner; tol=0.05) && npoints(learner) < 1000
    points = ask(learner, 1)
    isempty(points) && break
    for x in points
        tell!(learner, x, f(x))
    end
end

data = result(learner)   # (x = sorted Vector, y = values)
```

The full script is `examples/learner1d.jl`; run it from the repository root with
`julia --project=. examples/learner1d.jl`.

- `ask(learner, n)` returns up to `n` new points. The first points are the bounds; after that each
  point is the midpoint of the interval with the largest loss. The returned points are marked
  *pending*, so further calls never return them again, even before they are evaluated. Pass
  `tell_pending=false` to only peek.
- `tell!(learner, x, y)` records an evaluation. Points outside the bounds throw an `ArgumentError`;
  a repeated `x` is ignored and the first value is kept.
- `loss(learner)` is the largest interval loss, or `Inf` while a bound is unevaluated.
  `done(learner; tol)` is `loss(learner) <= tol`.
- `result(learner)` returns `(x, y)` copies sorted by `x`; `npoints(learner)` counts them.

## Refinement algorithm

Each interval between neighbouring evaluated points has a loss, by default the length of the
straight segment joining its endpoints, in coordinates scaled by the width of the bounds and the
observed range of `y`. This is upstream's `default_loss`. Steep regions have long segments and get
refined first. The scaled `y`-range is updated lazily: all losses are recomputed only when the range
more than doubles, as upstream does.

`ask` bisects the highest-loss interval, repeatedly. An interval that already contains pending
points is split into sub-intervals that each carry a share of its loss proportional to their width.
Ties go to the leftmost interval. Intervals narrower than about `eps` times the bounds have zero
loss and are never split, so `ask` returns fewer points (eventually none) once the resolution is
exhausted. A `NaN` or infinite interval loss is ranked by the interval's width instead.

## Deviations from Python Adaptive

These are intentional for now and may be revisited:

- `ask` returns only the points, not `(points, loss_improvements)`.
- `ask` bisects; upstream splits one interval into `k` equal parts when it receives several points.
- `tell!` with `x` outside the bounds throws; upstream stores the point and ignores it.
- `done(learner; tol)` exists; upstream expresses goals through runners (`loss_goal`).
- Not yet implemented: vector-valued `y`, neighbour-based losses (`nth_neighbors`, curvature),
  runners, `loss(real=false)`, optimized bulk `tell_many`.

## API

```@docs
AdaptivePort
Learner1D
ask
tell!
tell_pending!
loss
done
result
npoints
default_loss
```
