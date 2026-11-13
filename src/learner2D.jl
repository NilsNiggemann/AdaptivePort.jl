"""
    Learner2D{T, V, F, L}

Adaptive sampler of a scalar or vector-valued function on the rectangle `bounds = ((xmin, xmax), (ymin, ymax))`, driven by [`ask`](@ref) and [`tell!`](@ref).

Value type `V` is `T` for scalar output and `NTuple{M, T}` for vector output with `M = vdim > 1`. Fields: `f` (stored for future runners, never called), `bounds`, `loss_per_triangle`, `points`/`values` (evaluations in tell order), `index` (point → position in `points`) and `pending` (points handed out by [`ask`](@ref) but not yet told).

Invariants:
- `length(values) == length(points)`, `index[points[i]] == i`, `points` are distinct.
- `pending` is disjoint from `points`.
- All `points` and `pending` lie inside the closed rectangle `bounds`.

The learner works in scaled coordinates `(p - center) / (xmax - xmin, ymax - ymin)`, which map the domain to `[-1/2, 1/2]^2`, as upstream. Triangulations are rebuilt from scratch when needed, nothing geometric is cached.

Construct with [`Learner2D(f; bounds)`](@ref Learner2D(::Any)).

**See Also**
- [`ask`](@ref), [`tell!`](@ref), [`loss`](@ref), [`triangles`](@ref), [`default_loss`](@ref)
"""
struct Learner2D{T<:AbstractFloat, V, F, L} <: AbstractLearner
    f::F
    bounds::NTuple{2, NTuple{2, T}}
    loss_per_triangle::L
    points::Vector{NTuple{2, T}}
    values::Vector{V}
    index::Dict{NTuple{2, T}, Int}
    pending::Vector{NTuple{2, T}}
end

"""
    Learner2D(f; bounds, loss_per_triangle = default_loss, vdim = 1)

Create an empty learner for `f(x, y)` on the rectangle `bounds`. With `vdim = M > 1`, `f` returns `M` numbers and values are stored as `NTuple{M, T}`.

**Arguments**
- `f`: the function to learn; stored but never called by the learner.

**Keyword Arguments**
- `bounds`: `((xmin, xmax), (ymin, ymax))`, all finite with `xmin < xmax` and `ymin < ymax`; required because no sensible default domain exists. Integers are promoted to floating point.
- `loss_per_triangle`: function `(points, values, triangles) -> Vector` of per-triangle losses (see [`default_loss`](@ref)). `points` are in scaled coordinates, `values` are raw and of the stored type (`Vector{T}`, or `Vector{NTuple{M, T}}` for vector output), `triangles` are counter-clockwise index triples. Upstream passes a scipy interpolator holding the same three things.
- `vdim`: number of output components, default 1 (scalar values).

**Returns**
- A [`Learner2D`](@ref); throws `ArgumentError` for invalid bounds.

**See Also**
- [`ask`](@ref), [`tell!`](@ref)
"""
function Learner2D(f; bounds::NTuple{2, Tuple{Real, Real}}, loss_per_triangle = default_loss, vdim::Integer = 1)
    vdim >= 1 || throw(ArgumentError("vdim must be at least 1, got $vdim"))
    (xa, xb), (ya, yb) = bounds
    T = float(promote_type(typeof(xa), typeof(xb), typeof(ya), typeof(yb)))
    b = ((T(xa), T(xb)), (T(ya), T(yb)))
    if !(all(isfinite, (b[1]..., b[2]...)) && b[1][1] < b[1][2] && b[2][1] < b[2][2])
        throw(ArgumentError("bounds must be finite ((xmin, xmax), (ymin, ymax)) with xmin < xmax and ymin < ymax, got $bounds"))
    end
    P = NTuple{2, T}
    V = vdim == 1 ? T : NTuple{vdim, T}
    return Learner2D(f, b, loss_per_triangle, P[], V[], Dict{P, Int}(), P[])
end

npoints(l::Learner2D) = length(l.points)

"""
    result(learner::Learner2D)

Evaluated data as a named tuple `(points, values)` of copies in tell order: `points[i]` is an `(x, y)` tuple and `values[i]` its function value. [`triangles`](@ref) indexes into `points`.
"""
result(l::Learner2D) = (points = copy(l.points), values = copy(l.values))

# The four corners in upstream order (itertools.product of the bounds).
function corners(l::Learner2D)
    (xa, xb), (ya, yb) = l.bounds
    return ((xa, ya), (xa, yb), (xb, ya), (xb, yb))
end

corners_done(l::Learner2D) = all(c -> haskey(l.index, c), corners(l))

isknown(l::Learner2D, p) = haskey(l.index, p) || p in l.pending

function scale(l::Learner2D, p)
    (xa, xb), (ya, yb) = l.bounds
    return ((p[1] - (xa + xb) / 2) / (xb - xa), (p[2] - (ya + yb) / 2) / (yb - ya))
end

function unscale(l::Learner2D, p)
    (xa, xb), (ya, yb) = l.bounds
    return (p[1] * (xb - xa) + (xa + xb) / 2, p[2] * (yb - ya) + (ya + yb) / 2)
end

clip(l::Learner2D, p) = (clamp(p[1], l.bounds[1]...), clamp(p[2], l.bounds[2]...))

function to_point(l::Learner2D{T}, p) where {T}
    q = (T(p[1]), T(p[2]))
    if !(l.bounds[1][1] <= q[1] <= l.bounds[1][2] && l.bounds[2][1] <= q[2] <= l.bounds[2][2])
        throw(ArgumentError("point $p is outside the bounds $(l.bounds)"))
    end
    return q
end

isfinite_value(z::Real) = isfinite(z)
isfinite_value(z::Tuple) = all(isfinite, z)

zero_value(::Type{T}) where {T<:Real} = zero(T)
zero_value(::Type{NTuple{M, T}}) where {M, T} = ntuple(_ -> zero(T), M)

# Convert a told value to the stored type: `T` for scalar learners, `NTuple{M, T}` (any iterable of length `M`) for vector ones.
to_value(::Learner2D{T, T}, z::Real) where {T} = T(z)
function to_value(::Learner2D{T, NTuple{M, T}}, z) where {T, M}
    length(z) == M || throw(ArgumentError("expected a value with $M components, got $(length(z))"))
    return NTuple{M, T}(collect(z))
end

# Scaled points and values of the evaluated points with finite values: the vertices of the real triangulation.
function finite_data(l::Learner2D)
    idx = findall(isfinite_value, l.values)
    return idx, [scale(l, p) for p in view(l.points, idx)], l.values[idx]
end

"""
    triangles(learner::Learner2D)

Delaunay triangles of the evaluated points with finite values, as counter-clockwise index triples into `result(learner).points`.

The triangulation is computed in scaled coordinates (as the learner sees it), so it can differ from a Delaunay triangulation of the raw points when the domain is not square. Once the four corners are evaluated it covers the whole rectangle. Empty while fewer than three non-collinear points exist.

**See Also**
- [`result`](@ref), [`Learner2D`](@ref)
"""
function triangles(l::Learner2D)
    idx, pts, _ = finite_data(l)
    return [map(v -> idx[v], t) for t in delaunay_triangles(pts)]
end

function loss(l::Learner2D{T}) where {T}
    corners_done(l) || return T(Inf)
    _, pts, vals = finite_data(l)
    ts = delaunay_triangles(pts)
    isempty(ts) && return T(Inf)
    # ponytail: rebuilds the triangulation on every call, cache it per tell! if loss() shows up in profiles
    return T(maximum(l.loss_per_triangle(pts, vals, ts)))
end

function tell_pending!(l::Learner2D, p::Tuple{Real, Real})
    q = to_point(l, p)
    isknown(l, q) || push!(l.pending, q)
    return l
end

"""
    tell!(learner::Learner2D, (x, y), z)

Record the evaluation `z = f(x, y)`; for vector output (`vdim = M > 1`) `z` is any iterable of length `M`, otherwise `ArgumentError`.

Throws `ArgumentError` if `(x, y)` is outside the bounds (deviation: upstream stores but ignores such points). Telling an evaluated point again overwrites its value, as upstream. Non-finite `z` (any non-finite component) is stored but left out of the triangulation.
"""
function tell!(l::Learner2D, p::Tuple{Real, Real}, z)
    q = to_point(l, p)
    v = to_value(l, z)
    i = get(l.index, q, 0)
    if i == 0
        push!(l.points, q)
        push!(l.values, v)
        l.index[q] = length(l.points)
    else
        l.values[i] = v
    end
    j = findfirst(==(q), l.pending)
    j === nothing || deleteat!(l.pending, j)
    return l
end

"""
    ask(learner::Learner2D, n; tell_pending = true)

Propose up to `n` new points `(x, y)` to evaluate, as a `Vector`.

Corners that are neither evaluated nor pending come first. The rest follows upstream's `_fill_stack`: triangulate the evaluated points together with the pending and already chosen points (whose values are linearly interpolated from the evaluated data, or zero while a corner is unevaluated), rank the triangles by `loss_per_triangle`, and take one point from each of the highest-loss triangles (see `choose_point_in_triangle`). If that yields too few points, the triangulation is rebuilt with the new points and the step repeats. Candidates that coincide with a known point are skipped; fewer than `n` points (possibly none) are returned when no new candidate exists.

Deviation from upstream: no persistent point stack (`stack_size`); every call recomputes, and only the points are returned, not their loss improvements.

**Arguments**
- `learner`: the [`Learner2D`](@ref).
- `n`: number of points to request, `n >= 0`.

**Keyword Arguments**
- `tell_pending`: register the returned points via [`tell_pending!`](@ref). Default `true`, so that repeated calls never return the same point twice.
"""
function ask(l::Learner2D{T}, n::Integer; tell_pending = true) where {T}
    n < 0 && throw(ArgumentError("n must be non-negative, got $n"))
    points = NTuple{2, T}[]
    for c in corners(l)
        length(points) >= n && break
        isknown(l, c) || push!(points, c)
    end
    while length(points) < n
        choose_points!(points, l, n - length(points)) == 0 && break
    end
    if tell_pending
        for p in points
            tell_pending!(l, p)
        end
    end
    return points
end

# Scaled points and values of the evaluated, pending and `chosen` points (upstream `_data_combined`).
function combined_data(l::Learner2D, chosen)
    _, pts, vals = finite_data(l)
    extra = [scale(l, p) for p in Iterators.flatten((l.pending, chosen))]
    if corners_done(l) && length(pts) >= 3
        tri = delaunay(pts)
        extra_vals = interpolate_values(tri, vals, extra)
    else
        extra_vals = fill(zero_value(eltype(vals)), length(extra))
    end
    keep = isfinite_value.(extra_vals)  # outside the hull of the finite data (a corner with a NaN value)
    return vcat(pts, extra[keep]), vcat(vals, extra_vals[keep])
end

# Linear interpolation of `vals` (scalars or tuples, interpolated per component) at the points `qs`.
interpolate_values(tri, vals::Vector{<:Real}, qs) = [interpolate_linear(tri, vals, q) for q in qs]
function interpolate_values(tri, vals::Vector{<:Tuple}, qs)
    comps = ntuple(c -> map(v -> v[c], vals), length(first(vals)))
    return [map(vc -> interpolate_linear(tri, vc, q), comps) for q in qs]
end

# Append up to `m` new points to `chosen`, one per triangle in order of decreasing loss. Returns the number added.
function choose_points!(chosen, l::Learner2D, m)
    pts, vals = combined_data(l, chosen)
    ts = delaunay_triangles(pts)
    isempty(ts) && return 0
    # ponytail: full triangulation rebuild per call, O(N log N); use DT.add_point!/delete_point! if N ~ 1e5
    losses = l.loss_per_triangle(pts, vals, ts)
    added = 0
    for t in sortperm(losses; rev = true)
        added == m && break
        i, j, k = ts[t]
        p = clip(l, unscale(l, choose_point_in_triangle(pts[i], pts[j], pts[k])))
        if !isknown(l, p) && !(p in chosen)
            push!(chosen, p)
            added += 1
        end
    end
    return added
end

"""
    choose_point_in_triangle(a, b, c; max_badness = 5)

New sample point in the triangle `(a, b, c)`: the midpoint of its longest edge if the triangle is skinny, otherwise its centroid (upstream `choose_point_in_triangle`).

Badness is `sqrt(3) / 4 * longest_edge^2 / area`, 1 for an equilateral triangle; `max_badness = 5` is upstream's value. Splitting the longest edge of skinny triangles keeps the triangulation from accumulating slivers.
"""
function choose_point_in_triangle(a, b, c; max_badness = 5)
    edges = ((c, a), (a, b), (b, c))
    lengths = map(e -> hypot(e[2][1] - e[1][1], e[2][2] - e[1][2]), edges)
    i = argmax(lengths)
    badness = sqrt(3) / 4 * lengths[i]^2 / signed_area(a, b, c)
    if badness > max_badness
        p, q = edges[i]
        return ((p[1] + q[1]) / 2, (p[2] + q[2]) / 2)
    end
    return ((a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3)
end

"""
    default_loss(points, values, triangles)

Per-triangle loss `dev * sqrt(A) + 0.3 * A` of upstream's 2D `default_loss`: `A` is the triangle area and `dev` the [`deviations`](@ref AdaptivePort.deviations) estimate of the linear interpolation error.

The `dev` term refines where the function curves; the area term keeps refining flat regions so that no part of the domain is neglected. `points` are scaled to the unit square by the learner, so the loss is invariant under rescaling `x`, `y` or the values.

**Arguments**
- `points`: scaled vertex coordinates, `NTuple{2}` each.
- `values`: function values at `points`, `Vector{T}` or, for vector output, `Vector{NTuple{M, T}}` (losses are then summed over components, with values normalised by the largest component range, as upstream).
- `triangles`: counter-clockwise index triples into `points`.

**Returns**
- `Vector` of losses, one per triangle.

**See Also**
- [`Learner2D`](@ref)
"""
function default_loss(points, values, triangles)
    dev = deviations(points, values, triangles)
    return map(triangles, dev) do (i, j, k), d
        A = abs(signed_area(points[i], points[j], points[k]))
        d * sqrt(A) + 3A / 10
    end
end

"""
    deviations(points, values, triangles)

Per-triangle estimate of the linear interpolation error, as upstream `deviations`: with values normalised by their range and gradients `g` from [`estimate_gradients`](@ref AdaptivePort.estimate_gradients), `dev = sum_i max_j abs(v_i + g_i ⋅ (p_j - p_i) - v_j)` over the vertices `i`, `j` of the triangle. For vector values (`Vector{NTuple{M, T}}`) all components are normalised by the largest component range, gradients are estimated per component and the deviations are summed over components.
"""
function deviations(points, values::AbstractVector{<:Real}, triangles)
    lo, hi = extrema(values)
    return component_deviations(points, values ./ (hi > lo ? hi - lo : one(hi)), triangles)
end

function deviations(points, values::AbstractVector{<:Tuple}, triangles)
    comps = ntuple(c -> map(v -> v[c], values), length(first(values)))
    range = maximum(c -> (e = extrema(c); e[2] - e[1]), comps)
    return sum(c -> component_deviations(points, c ./ (range > 0 ? range : one(range)), triangles), comps)
end

# `v` is already normalised.
function component_deviations(points, v, triangles)
    g = estimate_gradients(points, v, triangles)
    return map(triangles) do t
        sum(t) do i
            maximum(t) do j
                abs(v[i] + g[i][1] * (points[j][1] - points[i][1]) + g[i][2] * (points[j][2] - points[i][2]) - v[j])
            end
        end
    end
end

"""
    estimate_gradients(points, values, triangles; tol = 1e-6, maxiter = 400)

Gradient of the data at every point, a port of scipy's `estimate_gradients_2d_global` (used by upstream through `CloughTocher2DInterpolator`).

The gradients minimise the curvature of the piecewise-cubic Clough–Tocher interpolant along the triangulation edges (G. M. Nielson, Math. Comp. 40, 253 (1983); R. J. Renka & A. K. Cline, Rocky Mountain J. Math. 14, 223 (1984)), solved by Gauss–Seidel iteration until the largest relative change is below `tol`. Points without neighbours get a zero gradient.
"""
function estimate_gradients(points, values, triangles; tol = 1e-6, maxiter = 400)
    T = eltype(values)
    neighbors = [Int[] for _ in eachindex(points)]
    for (i, j, k) in triangles, (a, b) in ((i, j), (j, k), (k, i))
        b in neighbors[a] || push!(neighbors[a], b)
        a in neighbors[b] || push!(neighbors[b], a)
    end
    g = fill((zero(T), zero(T)), length(points))
    for _ in 1:maxiter
        err = zero(T)
        for i in eachindex(points)
            isempty(neighbors[i]) && continue
            q11 = q12 = q22 = s1 = s2 = zero(T)
            for j in neighbors[i]
                ex, ey = points[j][1] - points[i][1], points[j][2] - points[i][2]
                L3 = hypot(ex, ey)^3
                df = -ex * g[j][1] - ey * g[j][2]
                c = (6 * (values[i] - values[j]) - 2 * df) / L3
                q11 += 4 * ex * ex / L3
                q12 += 4 * ex * ey / L3
                q22 += 4 * ey * ey / L3
                s1 += c * ex
                s2 += c * ey
            end
            det = q11 * q22 - q12^2
            r1 = (q22 * s1 - q12 * s2) / det
            r2 = (q11 * s2 - q12 * s1) / det
            err = max(err, max(abs(g[i][1] + r1), abs(g[i][2] + r2)) / max(one(T), abs(r1), abs(r2)))
            g[i] = (-r1, -r2)
        end
        err < tol && break
    end
    return g
end
