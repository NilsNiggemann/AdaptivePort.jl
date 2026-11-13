"""
    Learner1D{T, F, L}

Adaptive sampler of a scalar function on the interval `bounds`, driven by [`ask`](@ref) and [`tell!`](@ref).

Fields: `f` (stored for future runners, never called), `bounds`, `loss_per_interval`, `dx_eps` (smallest
splittable interval width, `2 * max(abs.(bounds)) * eps(T)` as upstream), `xs`/`ys` (evaluated points),
`losses` (cached loss per interval), `pending` (points handed out by [`ask`](@ref) but not yet told),
`yrange` (min and max of the finite told `ys`) and `yscale` (y-scale the cached `losses` were computed with).

Invariants:
- `xs` is strictly increasing, `length(ys) == length(xs)`.
- `length(losses) == max(0, length(xs) - 1)`, `losses[i] == interval_loss(learner, i)` for the interval `(xs[i], xs[i+1])`, evaluated with the current `yscale`; it is `zero(T)` if the width is below `dx_eps`.
- `yrange[2] - yrange[1] <= 2 * yscale` once two finite distinct `ys` exist: as upstream, all losses are recomputed only when the y-range more than doubles, so that `tell!` stays O(1) in loss evaluations.
- `pending` is strictly increasing and disjoint from `xs`.
- All `xs` and `pending` lie inside `bounds`.

Construct with [`Learner1D(f; bounds)`](@ref Learner1D).

**See Also**
- [`ask`](@ref), [`tell!`](@ref), [`loss`](@ref), [`default_loss`](@ref)
"""
mutable struct Learner1D{T<:AbstractFloat, F, L}
    f::F
    bounds::Tuple{T, T}
    loss_per_interval::L
    dx_eps::T
    xs::Vector{T}
    ys::Vector{T}
    losses::Vector{T}
    pending::Vector{T}
    yrange::Tuple{T, T}
    yscale::T
end

"""
    default_loss(xs, ys)

Loss of one interval: the length `hypot(x2 - x1, y2 - y1)` of the segment joining its endpoints, as upstream `default_loss`.

The learner passes *scaled* coordinates, `x / (b - a)` and `y / yscale`, so the loss is invariant under rescaling `x` or `y`. Measuring the graph's arc length makes steep regions dominate, while flat regions are still refined through the `x` term. A loss of the form `dx * (1 + abs(dy))` was tried first and rejected: for small `dx` the `dy` term vanishes and the sampling becomes uniform.

**Arguments**
- `xs`: the two scaled interval endpoints.
- `ys`: the two scaled function values.

**Returns**
- The interval loss, same type as the inputs.

**See Also**
- [`Learner1D`](@ref)
"""
default_loss(xs, ys) = hypot(xs[2] - xs[1], ys[2] - ys[1])

"""
    Learner1D(f; bounds, loss_per_interval = default_loss)

Create an empty learner for `f` on the interval `bounds`.

**Arguments**
- `f`: the function to learn; stored but never called by the learner.

**Keyword Arguments**
- `bounds`: `(a, b)` with both finite and `a < b`; required because no sensible default interval exists. Integers are promoted to floating point.
- `loss_per_interval`: function `(xs, ys) -> loss` of one interval in scaled coordinates (see [`default_loss`](@ref)), as upstream.

**Returns**
- A [`Learner1D`](@ref); throws `ArgumentError` for invalid bounds.

**See Also**
- [`ask`](@ref), [`tell!`](@ref)
"""
function Learner1D(f; bounds::Tuple{Real, Real}, loss_per_interval = default_loss)
    T = float(promote_type(typeof(bounds[1]), typeof(bounds[2])))
    a, b = T(bounds[1]), T(bounds[2])
    if !(isfinite(a) && isfinite(b) && a < b)
        throw(ArgumentError("bounds must be finite with a < b, got $bounds"))
    end
    dx_eps = 2 * max(abs(a), abs(b)) * eps(T)
    return Learner1D(f, (a, b), loss_per_interval, dx_eps, T[], T[], T[], T[], (T(Inf), T(-Inf)), zero(T))
end

"""
    npoints(learner)

Number of evaluated points of `learner`.

**See Also**
- [`result`](@ref)
"""
npoints(l::Learner1D) = length(l.xs)

"""
    result(learner)

Evaluated data of `learner` as a named tuple `(x, y)` of sorted copies.

**See Also**
- [`npoints`](@ref), [`tell!`](@ref)
"""
result(l::Learner1D) = (x = copy(l.xs), y = copy(l.ys))

xscale(l::Learner1D) = l.bounds[2] - l.bounds[1]

# Ranking priority of an interval of width dx: NaN/Inf losses fall back to the scaled width (upstream finite_loss).
priority(l::Learner1D, L, dx) = isfinite(L) ? L : dx / xscale(l)

# Real loss of the interval (xs[i], xs[i+1]) in scaled coordinates; zero below dx_eps (upstream behaviour).
function interval_loss(l::Learner1D, i)
    dx = l.xs[i+1] - l.xs[i]
    dx < l.dx_eps && return zero(dx)
    sx = xscale(l)
    sy = iszero(l.yscale) ? one(sx) : l.yscale
    return l.loss_per_interval((l.xs[i] / sx, l.xs[i+1] / sx), (l.ys[i] / sy, l.ys[i+1] / sy))
end

"""
    loss(learner)

Current loss of `learner`: the largest interval priority, or `Inf` while a bound is not yet evaluated.

Intervals whose loss is `NaN` or infinite are ranked by their width instead, like upstream.

**See Also**
- [`done`](@ref), [`default_loss`](@ref)
"""
function loss(l::Learner1D{T}) where {T}
    if isempty(l.xs) || l.xs[1] != l.bounds[1] || l.xs[end] != l.bounds[2]
        return T(Inf)
    end
    worst = zero(T)
    for i in eachindex(l.losses)
        worst = max(worst, priority(l, l.losses[i], l.xs[i+1] - l.xs[i]))
    end
    return worst
end

"""
    done(learner; tol)

Whether `loss(learner) <= tol`.

**Keyword Arguments**
- `tol`: loss tolerance; required because the loss scale depends on the loss function.

**See Also**
- [`loss`](@ref)
"""
done(l::Learner1D; tol) = loss(l) <= tol

function check_bounds(l::Learner1D, x)
    if !(l.bounds[1] <= x <= l.bounds[2])
        throw(ArgumentError("x = $x is outside the bounds $(l.bounds)"))
    end
    return nothing
end

"""
    tell_pending!(learner, x)

Mark `x` as issued but not yet evaluated, so [`ask`](@ref) does not hand it out again.

Throws `ArgumentError` if `x` is outside the bounds; a point that is already evaluated or pending is ignored.

**See Also**
- [`ask`](@ref), [`tell!`](@ref)
"""
function tell_pending!(l::Learner1D{T}, x::Real) where {T}
    xt = T(x)
    check_bounds(l, xt)
    i = searchsortedfirst(l.xs, xt)
    i <= length(l.xs) && l.xs[i] == xt && return l
    j = searchsortedfirst(l.pending, xt)
    j <= length(l.pending) && l.pending[j] == xt && return l
    # ponytail: O(N) insert into a Vector, use a sorted container if N reaches ~1e5
    insert!(l.pending, j, xt)
    return l
end

"""
    tell!(learner, x, y)

Record the evaluation `y = f(x)` and update the cached interval losses.

Throws `ArgumentError` if `x` is outside the bounds (deviation: upstream stores but ignores such points). A repeated `x` is silently ignored and the first value is kept, as upstream.

**See Also**
- [`ask`](@ref), [`tell_pending!`](@ref)
"""
function tell!(l::Learner1D{T}, x::Real, y::Real) where {T}
    xt, yt = T(x), T(y)
    check_bounds(l, xt)
    i = searchsortedfirst(l.xs, xt)
    i <= length(l.xs) && l.xs[i] == xt && return l
    j = searchsortedfirst(l.pending, xt)
    j <= length(l.pending) && l.pending[j] == xt && deleteat!(l.pending, j)
    # ponytail: O(N) insert into Vectors, use a sorted container if N reaches ~1e5
    insert!(l.xs, i, xt)
    insert!(l.ys, i, yt)
    n = length(l.xs)
    if isfinite(yt)
        l.yrange = (min(l.yrange[1], yt), max(l.yrange[2], yt))
    end
    n == 1 && return l
    insert!(l.losses, min(i, n - 1), zero(T))
    if l.yrange[2] - l.yrange[1] > 2 * l.yscale
        l.yscale = l.yrange[2] - l.yrange[1]
        for k in eachindex(l.losses)
            l.losses[k] = interval_loss(l, k)
        end
    else
        i > 1 && (l.losses[i-1] = interval_loss(l, i - 1))
        i < n && (l.losses[i] = interval_loss(l, i))
    end
    return l
end

"""
    ask(learner, n; tell_pending = true)

Propose up to `n` new points to evaluate, as a `Vector`.

Unevaluated and non-pending bounds come first (ascending). The rest is greedy bisection: the interval of `xs`, `pending` and chosen bounds with the highest priority is split at its midpoint, repeatedly. Pending points inside an evaluated interval take a proportional share of its loss, as upstream. Fewer than `n` points (possibly none) are returned once every interval is narrower than `dx_eps`.

Deviation from upstream (provisional): upstream returns `(points, loss_improvements)` and subdivides one interval into equal parts; this returns the points only.

**Arguments**
- `learner`: the [`Learner1D`](@ref).
- `n`: number of points to request, `n >= 0`.

**Keyword Arguments**
- `tell_pending`: register the returned points via [`tell_pending!`](@ref). Default `true`, so that repeated calls never return the same point twice.

**Returns**
- The proposed points; bounds first, then in the order chosen.

**See Also**
- [`tell!`](@ref), [`tell_pending!`](@ref)
"""
function ask(l::Learner1D{T}, n::Integer; tell_pending = true) where {T}
    n < 0 && throw(ArgumentError("n must be non-negative, got $n"))
    points = T[]
    for b in l.bounds
        length(points) >= n && break
        if !insorted(b, l.xs) && !insorted(b, l.pending)
            push!(points, b)
        end
    end
    if length(points) < n
        # Candidate intervals (ls[k], rs[k]) of the combined point set with their priorities.
        combined = sort!(vcat(l.xs, l.pending, points))
        ls = combined[1:end-1]
        rs = combined[2:end]
        prios = similar(ls)
        for k in eachindex(ls)
            j = searchsortedlast(l.xs, ls[k])
            if 1 <= j < length(l.xs) && rs[k] <= l.xs[j+1]
                dx = l.xs[j+1] - l.xs[j]
                prios[k] = priority(l, l.losses[j], dx) * (rs[k] - ls[k]) / dx
            else
                prios[k] = (rs[k] - ls[k]) / xscale(l)
            end
        end
        while length(points) < n
            # ponytail: O(n*m) scan for the best interval, use a heap if n*m gets large
            best = 0
            for k in eachindex(ls)
                m = ls[k] + (rs[k] - ls[k]) / 2
                splittable = rs[k] - ls[k] >= l.dx_eps && ls[k] < m < rs[k]
                if splittable && (best == 0 || prios[k] > prios[best])
                    best = k
                end
            end
            best == 0 && break
            m = ls[best] + (rs[best] - ls[best]) / 2
            push!(points, m)
            prios[best] /= 2
            insert!(ls, best + 1, m)
            insert!(rs, best + 1, rs[best])
            insert!(prios, best + 1, prios[best])
            rs[best] = m
        end
    end
    if tell_pending
        for x in points
            tell_pending!(l, x)
        end
    end
    return points
end
