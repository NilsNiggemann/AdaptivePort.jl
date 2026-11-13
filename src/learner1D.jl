"""
    Learner1D{T, V, F, L}

Adaptive sampler of a scalar or vector-valued function on the interval `bounds`, driven by [`ask`](@ref) and [`tell!`](@ref).

The value type `V` is `T` for scalar output (`vdim = 1`) and `NTuple{vdim, T}` for vector output. Vector values are
scaled by one common y-scale, the largest per-component range, as upstream. Non-finite components are ignored when
tracking the range.

Fields: `f` (stored for future runners, never called), `bounds`, `loss_per_interval`, `dx_eps` (smallest
splittable interval width, `2 * max(abs.(bounds)) * eps(T)` as upstream), `xs`/`ys` (evaluated points),
`losses` (cached loss per interval), `pending` (points handed out by [`ask`](@ref) but not yet told),
`yrange` (min and max of the finite told `ys`, per component for vector output) and `yscale` (the y-range at the last full recomputation of `losses`).

Invariants:
- `xs` is strictly increasing, `length(ys) == length(xs)`.
- `length(losses) == max(0, length(xs) - 1)`, `losses[i]` is the loss of the interval `(xs[i], xs[i+1])`, evaluated with the y-range current when it was last computed (as upstream, losses of untouched intervals are not refreshed until the range more than doubles); it is `zero(T)` if the width is below `dx_eps`.
- `maximum(yrange[2] .- yrange[1]) <= 2 * yscale` once two finite distinct `ys` exist: as upstream, all losses are recomputed only when the y-range more than doubles, so that `tell!` stays O(1) in loss evaluations.
- `pending` is strictly increasing and disjoint from `xs`.
- All `xs` and `pending` lie inside `bounds`.

Construct with [`Learner1D(f; bounds)`](@ref Learner1D).

**See Also**
- [`ask`](@ref), [`tell!`](@ref), [`loss`](@ref), [`default_loss`](@ref)
"""
mutable struct Learner1D{T<:AbstractFloat, V, F, L} <: AbstractLearner
    f::F
    bounds::Tuple{T, T}
    loss_per_interval::L
    dx_eps::T
    xs::Vector{T}
    ys::Vector{V}
    losses::Vector{T}
    pending::Vector{T}
    yrange::Tuple{V, V}
    yscale::T
end

"""
    default_loss(xs, ys)

Loss of one interval: the length `hypot(x2 - x1, y2 - y1)` of the segment joining its endpoints, as upstream `default_loss`.

For vector values (a tuple of two `NTuple`s) it is the maximum over the components of `hypot(dx, abs(dy_i))`, as upstream; a `NaN` component makes the loss `NaN`.

The learner passes *scaled* coordinates, `x / (b - a)` and `y / yscale`, so the loss is invariant under rescaling `x` or `y`. Measuring the graph's arc length makes steep regions dominate, while flat regions are still refined through the `x` term. A loss of the form `dx * (1 + abs(dy))` was tried first and rejected: for small `dx` the `dy` term vanishes and the sampling becomes uniform.

**Arguments**
- `xs`: the two scaled interval endpoints.
- `ys`: the two scaled function values; for vector output two `NTuple`s, scaled by the common y-scale.

**Returns**
- The interval loss, same type as the inputs.

**See Also**
- [`Learner1D`](@ref)
"""
default_loss(xs, ys) = hypot(xs[2] - xs[1], ys[2] - ys[1])

function default_loss(xs, ys::Tuple{NTuple, NTuple})
    dx = xs[2] - xs[1]
    return maximum(dy -> hypot(dx, dy), abs.(ys[2] .- ys[1]))
end

"""
    Learner1D(f; bounds, loss_per_interval = default_loss, vdim = 1)

Create an empty learner for `f` on the interval `bounds`.

**Arguments**
- `f`: the function to learn; stored but never called by the learner.

**Keyword Arguments**
- `bounds`: `(a, b)` with both finite and `a < b`; required because no sensible default interval exists. Integers are promoted to floating point.
- `loss_per_interval`: function `(xs, ys) -> loss` of one interval in scaled coordinates (see [`default_loss`](@ref)), as upstream. For `vdim > 1` it receives two `NTuple{vdim}` of scaled values instead of two numbers.
- `vdim`: length of the function's output. `1` (default) stores plain numbers; `M > 1` stores `NTuple{M, T}` and [`tell!`](@ref) accepts any iterable of length `M`.

**Returns**
- A [`Learner1D`](@ref); throws `ArgumentError` for invalid bounds.

**See Also**
- [`ask`](@ref), [`tell!`](@ref)
"""
function Learner1D(f; bounds::Tuple{Real, Real}, loss_per_interval = default_loss, vdim::Integer = 1)
    T = float(promote_type(typeof(bounds[1]), typeof(bounds[2])))
    a, b = T(bounds[1]), T(bounds[2])
    if !(isfinite(a) && isfinite(b) && a < b)
        throw(ArgumentError("bounds must be finite with a < b, got $bounds"))
    end
    dx_eps = 2 * max(abs(a), abs(b)) * eps(T)
    vdim >= 1 || throw(ArgumentError("vdim must be at least 1, got $vdim"))
    if vdim == 1
        return Learner1D(f, (a, b), loss_per_interval, dx_eps, T[], T[], T[], T[], (T(Inf), T(-Inf)), zero(T))
    end
    V = NTuple{vdim, T}
    yrange = (ntuple(_ -> T(Inf), vdim), ntuple(_ -> T(-Inf), vdim))
    return Learner1D(f, (a, b), loss_per_interval, dx_eps, T[], V[], T[], T[], yrange, zero(T))
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

Evaluated data of `learner` as a named tuple `(x, y)` of sorted copies; `y` holds tuples for vector output.

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
    sy = current_yscale(l)
    return l.loss_per_interval((l.xs[i] / sx, l.xs[i+1] / sx), (l.ys[i] ./ sy, l.ys[i+1] ./ sy))
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

pending_points(l::Learner1D) = copy(l.pending)

function remove_unfinished!(l::Learner1D)
    empty!(l.pending)
    return l
end

"""
    tell!(learner, x, y)

Record the evaluation `y = f(x)` and update the cached interval losses.

For vector output (`vdim > 1`) `y` is any iterable of length `vdim`; other lengths throw `ArgumentError`. Throws `ArgumentError` if `x` is outside the bounds (deviation: upstream stores but ignores such points). A repeated `x` is silently ignored and the first value is kept, as upstream; use `update_value!` internally to overwrite.

**See Also**
- [`ask`](@ref), [`tell_pending!`](@ref)
"""
tell!(l::Learner1D{T,T}, x::Real, y::Real) where {T} = insert_point!(l, T(x), T(y))

function tell!(l::Learner1D{T,V}, x::Real, y) where {T, V<:Tuple}
    return insert_point!(l, T(x), as_value(V, y))
end

# Convert any iterable of the right length to the stored tuple type.
function as_value(::Type{NTuple{M,T}}, y) where {M, T}
    t = Tuple(y)
    length(t) == M || throw(ArgumentError("expected $M output components, got $(length(t))"))
    return NTuple{M,T}(t)
end

# Widen the y-range by the finite components of y (upstream never shrinks it).
function widen_range(range, y)
    lo = ifelse.(isfinite.(y), min.(range[1], y), range[1])
    hi = ifelse.(isfinite.(y), max.(range[2], y), range[2])
    return (lo, hi)
end

spread(range) = maximum(range[2] .- range[1])

# y-scale of newly computed losses: the current range, 1 while it is zero or empty (upstream `_scale_y`).
function current_yscale(l::Learner1D)
    s = spread(l.yrange)
    return s > 0 ? s : one(s)
end

function insert_point!(l::Learner1D{T}, xt::T, yt) where {T}
    check_bounds(l, xt)
    i = searchsortedfirst(l.xs, xt)
    i <= length(l.xs) && l.xs[i] == xt && return l
    j = searchsortedfirst(l.pending, xt)
    j <= length(l.pending) && l.pending[j] == xt && deleteat!(l.pending, j)
    # ponytail: O(N) insert into Vectors, use a sorted container if N reaches ~1e5
    insert!(l.xs, i, xt)
    insert!(l.ys, i, yt)
    l.yrange = widen_range(l.yrange, yt)
    n = length(l.xs)
    n == 1 && return l
    insert!(l.losses, min(i, n - 1), zero(T))
    refresh_losses!(l, i)
    return l
end

# Recompute the losses touching point i, or all of them when the y-range more than doubled (upstream).
function refresh_losses!(l::Learner1D, i)
    n = length(l.xs)
    if spread(l.yrange) > 2 * l.yscale
        l.yscale = spread(l.yrange)
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
    update_value!(learner, x, y)

Overwrite the value at the already evaluated point `x` and update the neighbouring losses.

Unexported; used by `AverageLearner1D` to move a mean. Throws `ArgumentError` if `x` is not evaluated. Values are converted as in [`tell!`](@ref). As upstream (whose `AverageLearner1D` only calls `_update_scale`, which widens), the y-range never shrinks, so an overwritten extreme keeps counting; all losses are recomputed only when the range more than doubles.

**See Also**
- [`tell!`](@ref)
"""
update_value!(l::Learner1D{T,T}, x::Real, y::Real) where {T} = overwrite_point!(l, T(x), T(y))

function update_value!(l::Learner1D{T,V}, x::Real, y) where {T, V<:Tuple}
    return overwrite_point!(l, T(x), as_value(V, y))
end

function overwrite_point!(l::Learner1D, xt, yt)
    i = searchsortedfirst(l.xs, xt)
    if i > length(l.xs) || l.xs[i] != xt
        throw(ArgumentError("x = $xt has not been evaluated"))
    end
    l.ys[i] = yt
    l.yrange = widen_range(l.yrange, yt)
    length(l.xs) > 1 && refresh_losses!(l, i)
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
