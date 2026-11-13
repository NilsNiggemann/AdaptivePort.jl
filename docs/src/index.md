# AdaptivePort.jl

Julia port of [Python Adaptive](https://github.com/python-adaptive/adaptive): sample a function
where it is most informative instead of on a fixed grid. Provides [`Learner1D`](@ref) for scalar
functions on an interval and [`Learner2D`](@ref) for scalar functions on a rectangle. Both implement
the [`AbstractLearner`](@ref) interface.

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

## Learner2D

```julia
using AdaptivePort

f(x, y) = sin(5x) * cos(7y) + 0.3 * exp(-20 * ((x - 0.3)^2 + (y + 0.2)^2))
learner = Learner2D(f; bounds=((-1, 1), (-0.75, 0.75)))

while !done(learner; tol=0.01) && npoints(learner) < 2000
    points = ask(learner, 5)
    isempty(points) && break
    for (x, y) in points
        tell!(learner, (x, y), f(x, y))
    end
end

data = result(learner)   # (points = Vector of (x, y) tuples, values = Vector), in tell order
tris = triangles(learner) # counter-clockwise index triples into data.points
```

The full script is `examples/learner2d.jl`.

- Points are `NTuple{2}` tuples `(x, y)` everywhere: `ask` returns them, `tell!` takes them.
- `bounds = ((xmin, xmax), (ymin, ymax))`, finite with `xmin < xmax`, `ymin < ymax`; otherwise
  `ArgumentError`. Any aspect ratio and coordinate scale works: the learner works in scaled
  coordinates where the domain is $[-1/2, 1/2]^2$, as upstream.
- The first four points are the corners, in the order $(x_0, y_0)$, $(x_0, y_1)$, $(x_1, y_0)$,
  $(x_1, y_1)$. Their convex hull is the whole rectangle, so once they are evaluated the
  triangulation covers the domain and no boundary region can be neglected.
- `tell!(learner, (x, y), z)` throws for points outside the bounds. Telling a point again
  overwrites its value (as upstream; `Learner1D` keeps the first value, also as upstream). A
  non-finite `z` is stored but left out of the triangulation.
- `loss(learner)` is `Inf` until the four corners are evaluated, then the largest triangle loss
  (below). `done(learner; tol)` is `loss(learner) <= tol`.
- `result(learner)` returns copies `(points, values)` in tell order; `triangles(learner)` returns
  the Delaunay triangles of the evaluated finite-valued points as index triples into `points`.

### Triangulation

The Delaunay triangulation of the (scaled) points is built with
[DelaunayTriangulation.jl](https://github.com/JuliaGeometry/DelaunayTriangulation.jl), rebuilt from
scratch whenever it is needed (about 15 ms per `ask` at 1000 points). `src/triangulation.jl` is
the only file that uses the package.

### Loss and refinement

The loss of a triangle with area $A$ (in scaled coordinates) is upstream's `default_loss`,
$L = d\sqrt{A} + 0.3A$. Here $d$ estimates the error of linear interpolation on the triangle:
with values $v_i$ normalised by their total range and gradients $g_i$ estimated at the vertices,
$d = \sum_i \max_j \lvert v_i + g_i \cdot (p_j - p_i) - v_j \rvert$ over the vertices $i, j$. The
gradients are a port of scipy's `estimate_gradients_2d_global` (which upstream uses), checked
against scipy to $10^{-14}$. The $d$ term refines curved regions; the $0.3A$ term keeps refining
flat ones. Because coordinates and values are normalised, the loss is a relative quantity,
invariant under rescaling $x$, $y$ or $f$; `tol` in `done` is in these units. It decreases as
points are added, though not monotonically: adding a point changes the gradient estimates and
the value range.

`ask` triangulates the evaluated points together with the pending ones (whose values are linearly
interpolated from the evaluated data, so they add no deviation but split triangles), ranks the
triangles by loss and takes one new point from each of the `n` worst. In a triangle whose
badness $\sqrt{3}\ell^2/(4A)$ ($\ell$ the longest edge) exceeds 5 that point is the midpoint of the
longest edge, otherwise the centroid. If too few distinct points come out, the triangulation is
rebuilt including them and the step repeats.

## Deviations from Python Adaptive

These are intentional for now and may be revisited:

- `ask` returns only the points, not `(points, loss_improvements)`.
- `ask` bisects; upstream splits one interval into `k` equal parts when it receives several points.
- `tell!` with `x` outside the bounds throws; upstream stores the point and ignores it.
- `done(learner; tol)` exists; upstream expresses goals through runners (`loss_goal`).
- Not yet implemented: vector-valued `y`, neighbour-based losses (`nth_neighbors`, curvature),
  runners, `loss(real=false)`, optimized bulk `tell_many`.

`Learner2D` additionally:

- has no persistent point stack (`stack_size`): every `ask` recomputes the triangle losses, which
  equals upstream with `stack_size = 1`. Upstream's default (10) hands out points chosen from an
  older triangulation.
- may pick a different diagonal for cocircular points (e.g. the four corners), so point sequences
  differ from upstream after the corners, while the losses reached are comparable.
- ships only `default_loss`; `uniform_loss`, `resolution_loss_function`,
  `minimize_triangle_surface_loss` and `thresholded_loss_function` are not ported (pass any
  `(points, values, triangles) -> losses` function as `loss_per_triangle`).
- supports scalar values only, and does not provide an interpolator (`ip()`) or
  `interpolated_on_grid`.
- ignores non-finite values in the triangulation; upstream propagates them into the loss.

## API

```@docs
AdaptivePort
AbstractLearner
Learner1D
Learner2D
ask
tell!
tell_pending!
loss
done
result
npoints
default_loss
triangles
```

### Internals

```@docs
AdaptivePort.deviations
AdaptivePort.estimate_gradients
AdaptivePort.choose_point_in_triangle
AdaptivePort.delaunay
AdaptivePort.delaunay_triangles
AdaptivePort.signed_area
AdaptivePort.interpolate_linear
```
