# Doubly adaptive Clenshaw-Curtis quadrature, a port of upstream's `IntegratorLearner` (Pedro Gonnet's algorithm 4:
# "Increasing the Reliability of Adaptive Quadrature Using Explicit Interpolants", ACM TOMS 37 (3), 2008).
# Interval depths are 0-based as upstream (depth `d` has `Coeffs.NS[d + 1]` points); arrays are 1-based.

"""
    DivergentIntegralError()

Thrown by [`tell!`](@ref) when an [`IntegratorLearner`](@ref) detects a divergent integral: the integrand keeps growing
by more than a factor of two under repeated bisection, so the integral probably does not exist.
"""
struct DivergentIntegralError <: Exception end

Base.showerror(io::IO, ::DivergentIntegralError) = print(io, "DivergentIntegralError: the integral appears to be divergent")

# Curried identity test, for finding intervals in vectors.
same(x) = y -> y === x

# One interval of the quadrature tree. `data` holds the told values at its points, `c` the Legendre coefficients of
# the interpolant, `done_leaves` the descendants (or itself) that currently contribute to the integral estimate
# (`nothing`: already handed on to the ancestors), `depth_complete` the highest depth whose values are all known.
mutable struct Interval
    a::Float64
    b::Float64
    c::Vector{Float64}
    c00::Float64
    depth::Int
    igral::Float64
    err::Float64
    rdepth::Int
    ndiv::Int
    parent::Union{Interval,Nothing}
    children::Vector{Interval}
    data::Dict{Float64,Float64}
    done_leaves::Union{Vector{Interval},Nothing}
    depth_complete::Union{Int,Nothing}
    removed::Bool
end

Interval(a, b, depth, rdepth) = Interval(a, b, Float64[], 0.0, depth, 0.0, 0.0, rdepth, 0, nothing, Interval[], Dict{Float64,Float64}(), Interval[], nothing, false)

# err = floatmax (not Inf) because Inf / 2 == Inf, as upstream.
function make_first(a, b)
    iv = Interval(a, b, 2, 1)
    iv.err = floatmax(Float64)
    return iv
end

points(iv::Interval, depth = iv.depth) = (iv.a + iv.b) / 2 .+ (iv.b - iv.a) .* Coeffs.XI[depth+1] ./ 2

function refinement_complete(iv::Interval, depth)
    length(iv.data) < Coeffs.NS[depth+1] && return false
    return all(x -> haskey(iv.data, x), points(iv, depth))
end

# Shift matrix of a child interval; only for children of a split interval.
shift_matrix(iv::Interval) = iv.a == iv.parent.a ? Coeffs.T_LEFT : Coeffs.T_RIGHT

function split!(iv::Interval)
    pts = points(iv)
    m = pts[length(pts)÷2+1]
    ivals = [Interval(iv.a, m, 0, iv.rdepth + 1), Interval(m, iv.b, 0, iv.rdepth + 1)]
    iv.children = ivals
    for child in ivals
        child.parent = iv
        child.ndiv = iv.ndiv
        child.err = iv.err / 2
    end
    return ivals
end

calc_igral!(iv::Interval) = iv.igral = (iv.b - iv.a) * iv.c[1] / sqrt(2)

# Heuristic error (half the parent's) for an interval whose actual error is not yet computable, handed down.
function update_heuristic_err!(iv::Interval, value)
    iv.err = value
    for child in iv.children
        dc = child.depth_complete
        dc !== nothing && (dc != 0 || iv.depth_complete !== nothing) && continue
        update_heuristic_err!(child, value / 2)
    end
end

# Error from the difference of two coefficient vectors; returns the (unscaled) norm of the difference.
function calc_err!(iv::Interval, c_old)
    c_new = iv.c
    c_diff = zeros(max(length(c_old), length(c_new)))
    c_diff[eachindex(c_old)] .= c_old
    c_diff[eachindex(c_new)] .-= c_new
    nrm = LinearAlgebra.norm(c_diff)
    iv.err = (iv.b - iv.a) * nrm
    for child in iv.children
        child.depth_complete === nothing && update_heuristic_err!(child, iv.err / 2)
    end
    return nrm
end

function update_ndiv_recursively!(iv::Interval)
    iv.ndiv += 1
    iv.ndiv > Coeffs.NDIV_MAX && 2 * iv.ndiv > iv.rdepth && throw(DivergentIntegralError())
    foreach(update_ndiv_recursively!, iv.children)
end

function calc_ndiv!(iv::Interval)
    div = iv.parent.c00 != 0 && iv.c00 / iv.parent.c00 > 2
    iv.ndiv += div
    iv.ndiv > Coeffs.NDIV_MAX && 2 * iv.ndiv > iv.rdepth && throw(DivergentIntegralError())
    div && foreach(update_ndiv_recursively!, iv.children)
end

# Downdate of the coefficients `c` for the nodes `nans` (algorithm 5 of Gonnet's thesis): the interpolant of the
# remaining nodes, in place. Port of upstream's `_downdate` with every index shifted by one.
function downdate!(c, nans, depth)
    b = copy(Coeffs.B_DEF[depth+1])
    m = Coeffs.NS[depth+1] - 1
    for i in nans
        b[m+2] /= Coeffs.ALPHA[m+1]
        xii = Coeffs.XI[depth+1][i]
        b[m+1] = (b[m+1] + xii * b[m+2]) / Coeffs.ALPHA[m]
        for j in m-1:-1:1
            b[j+1] = (b[j+1] + xii * b[j+2] - Coeffs.GAMMA[j+2] * b[j+3]) / Coeffs.ALPHA[j]
        end
        popfirst!(b)
        factor = c[m+1] / b[m+1]
        for k in 1:m
            c[k] -= factor * b[k]
        end
        c[m+1] = 0
        m -= 1
    end
    return c
end

# Legendre coefficients of the interpolant through `fx`; non-finite values are removed by a downdate.
function calc_coeffs(fx, depth)
    nans = findall(!isfinite, fx)
    isempty(nans) && return Coeffs.V_INV[depth+1] * fx
    fx0 = copy(fx)
    fx0[nans] .= 0.0
    return downdate!(Coeffs.V_INV[depth+1] * fx0, nans, depth)
end

# Fold the interval into the integral estimate of its ancestors while all their children are known.
function propagate_done_leaves!(iv::Interval)
    iv.done_leaves = [iv]
    ival = iv.parent
    old_leaves = Interval[]
    while ival !== nothing
        unused = filter(ch -> ch.done_leaves !== nothing, ival.children)
        all(ch -> !isempty(ch.done_leaves), unused) || break
        ival.done_leaves === nothing && (ival.done_leaves = Interval[])
        push!(old_leaves, ival)
        for ch in ival.children
            ch.done_leaves === nothing && continue
            for leaf in ch.done_leaves
                any(same(leaf), ival.done_leaves) || push!(ival.done_leaves, leaf)
            end
            ch.done_leaves = nothing
        end
        filter!(leaf -> !any(same(leaf), old_leaves), ival.done_leaves)
        ival = ival.parent
    end
end

# Integral and error of `iv` from the complete values at `depth`. Returns `(force_split, remove)`.
function complete_process!(iv::Interval, depth)
    iv.depth_complete = depth
    fx = [iv.data[x] for x in points(iv, depth)]
    first_ival = iv.parent === nothing && depth == 2
    c_old = depth > 0 && !first_ival ? iv.c : Float64[]
    iv.c = calc_coeffs(fx, depth)
    if first_ival
        iv.c00 = 0.0
        return false, false
    end
    calc_igral!(iv)
    force_split = false
    if depth > 0
        c_diff = calc_err!(iv, c_old)
        force_split = c_diff > Coeffs.HINT * LinearAlgebra.norm(iv.c)
    else
        iv.c00 = iv.c[1]
        p = iv.parent
        if p.depth_complete !== nothing
            calc_err!(iv, shift_matrix(iv)[:, 1:Coeffs.NS[p.depth_complete+1]] * p.c)
            calc_ndiv!(iv)
        end
        for child in iv.children
            child.depth_complete !== nothing && calc_ndiv!(child)
            child.depth_complete == 0 && calc_err!(child, shift_matrix(child)[:, 1:Coeffs.NS[iv.depth_complete+1]] * iv.c)
        end
    end
    iv.done_leaves !== nothing && isempty(iv.done_leaves) && propagate_done_leaves!(iv)
    remove = iv.err < abs(iv.igral) * Coeffs.EPS * Coeffs.VCOND[depth+1]
    return force_split, remove
end

"""
    IntegratorLearner{F}

Adaptive quadrature of a scalar function on `bounds`, driven by [`ask`](@ref) and [`tell!`](@ref): Gonnet's
doubly adaptive Clenshaw-Curtis scheme (algorithm 4), as upstream.

Each interval holds a Clenshaw-Curtis rule of 5, 9, 17 or 33 nodes. The interval with the largest error estimate (the
L2 distance of two successive interpolants, times the width) is refined to the next rule, or bisected when it holds
33 nodes or when refining did not reduce the error. Values may be told in any order and some may be skipped; the
estimates use the intervals whose points are all known. Non-finite values are removed from the interpolant
(downdating) rather than failing.

Fields: `f` (called by runners), `bounds`, `tol` (relative tolerance for [`done`](@ref)), `max_ivals` (most intervals
kept, the one with the smallest error is dropped beyond it), `data` (told values), and the internal bookkeeping
of the interval tree (`pending`, `stack`, `x_mapping`, `ivals`, `priority_split`, `first_ival`).

Construct with [`IntegratorLearner(f; bounds, tol)`](@ref IntegratorLearner). The points are plain numbers `x`.

**See Also**
- [`DivergentIntegralError`](@ref), [`result`](@ref), [`done`](@ref)
"""
mutable struct IntegratorLearner{F} <: AbstractLearner
    f::F
    bounds::Tuple{Float64,Float64}
    tol::Float64
    max_ivals::Int
    priority_split::Vector{Interval}
    data::Dict{Float64,Float64}
    pending::Set{Float64}
    stack::Vector{Float64}
    x_mapping::Dict{Float64,Vector{Interval}}
    ivals::Vector{Interval}
    first_ival::Interval
end

"""
    IntegratorLearner(f; bounds = (-1, 1), tol = 1e-8)

Learner of the integral of `f` over `bounds`; runners call `f(x)`.

Always Float64, as the coefficient matrices are Float64 constants.

**Keyword Arguments**
- `bounds`: the integration interval.
- `tol`: relative tolerance; [`done`](@ref) holds once `error < tol * abs(integral)`, or when the remaining error comes
  only from intervals that cannot be refined further. 1e-8 because it is reached quickly for smooth functions; the
  estimate cannot go below machine precision anyway.

**Returns**
- An [`IntegratorLearner`](@ref) with the points of its first interval queued.

**See Also**
- [`BlockingRunner`](@ref), which stops on [`done`](@ref) without a `tol` for this learner
"""
function IntegratorLearner(f; bounds = (-1, 1), tol = 1e-8)
    a, b = Float64.(bounds)
    first_ival = make_first(a, b)
    l = IntegratorLearner(f, (a, b), Float64(tol), 1000, Interval[], Dict{Float64,Float64}(), Set{Float64}(), Float64[], Dict{Float64,Vector{Interval}}(), Interval[], first_ival)
    add_ival!(l, first_ival)
    return l
end

has_own_goal(::IntegratorLearner) = true

# Register an interval with the learner: map its points to it, queue the unknown ones, process the known ones.
function add_ival!(l::IntegratorLearner, iv::Interval)
    for x in points(iv)
        # x_mapping vectors are ordered by rdepth (stable), as upstream's SortedSet
        ivs = get!(Vector{Interval}, l.x_mapping, x)
        if !any(same(iv), ivs)
            insert!(ivs, something(findfirst(other -> other.rdepth > iv.rdepth, ivs), length(ivs) + 1), iv)
        end
        if haskey(l.data, x)
            tell!(l, x, l.data[x])
        elseif x ∉ l.pending
            push!(l.pending, x)
            push!(l.stack, x)
        end
    end
    # ponytail: linear membership test on a vector (at most max_ivals entries); a Set of identity-hashed keys breaks under deepcopy, so keep the vector
    any(same(iv), l.ivals) || push!(l.ivals, iv)
    return l
end

function propagate_removed!(l::IntegratorLearner, iv::Interval)
    iv.removed = true
    i = findfirst(same(iv), l.ivals)
    i === nothing || deleteat!(l.ivals, i)
    for child in iv.children
        propagate_removed!(l, child)
    end
end

"""
    tell!(learner::IntegratorLearner, x, y)

Record `y = f(x)` for a point `x` returned by [`ask`](@ref), in any order. Intervals whose values are complete update
the integral and error estimate. A non-finite `y` is allowed.

**Throws**
- `ArgumentError` if `x` was not issued by the learner.
- [`DivergentIntegralError`](@ref) if the integral appears to diverge.

**Returns**
- `learner`.
"""
function tell!(l::IntegratorLearner, x::Real, y::Real)
    x = Float64(x)
    haskey(l.x_mapping, x) || throw(ArgumentError("Point $x doesn't belong to any interval"))
    l.data[x] = y
    delete!(l.pending, x)
    # a told point is no longer queued: with `ask(...; tell_pending = false)` it was read from the queue, not popped
    i = findfirst(==(x), l.stack)
    i === nothing || deleteat!(l.stack, i)
    for iv in l.x_mapping[x]
        iv.data[x] = y
        from_depth = iv.depth_complete === nothing ? (iv.parent === nothing ? 2 : 0) : iv.depth_complete + 1
        for depth in from_depth:iv.depth
            refinement_complete(iv, depth) || continue
            force_split, remove = complete_process!(iv, depth)
            if remove
                # too narrow, or the error is at the limit of machine accuracy: drop it, remembering its error as excess
                propagate_removed!(l, iv)
            elseif force_split && isempty(iv.children)
                push!(l.priority_split, iv)
            end
        end
    end
    return l
end

"""
    tell_pending!(learner::IntegratorLearner, x)

Does nothing, as upstream: the learner tracks the points it hands out itself. Returns `learner`.
"""
tell_pending!(l::IntegratorLearner, x) = l

"""
    remove_unfinished!(learner::IntegratorLearner)

Does nothing, as upstream (its `remove_unfinished` is `pass`): points that were asked and never told stay pending and are
not asked again. Returns `learner`.
"""
remove_unfinished!(l::IntegratorLearner) = l

"""
    pending_points(learner::IntegratorLearner)

Sorted points asked from `learner` and not yet told. Points queued but not yet asked are not included.
"""
function pending_points(l::IntegratorLearner)
    queued = Set(l.stack)
    return sort!([x for x in l.pending if x ∉ queued])
end

improvement(l::IntegratorLearner, x) = maximum(iv.err for iv in l.x_mapping[x])

function pop_from_stack!(l::IntegratorLearner, n)
    xs = l.stack[1:min(n, end)]
    deleteat!(l.stack, 1:length(xs))
    return xs, Float64[improvement(l, x) for x in xs]
end

# Refine or split the interval with the largest error (or one that asked to be split), queueing new points; `false` if no interval is left.
function fill_stack!(l::IntegratorLearner)
    force_split = !isempty(l.priority_split)
    if force_split
        iv = pop!(l.priority_split)
    else
        isempty(l.ivals) && return false
        iv = l.ivals[1]
        for other in l.ivals
            (other.err, other.a) > (iv.err, iv.a) && (iv = other)
        end
    end
    i = findfirst(same(iv), l.ivals)
    i === nothing && return true
    pts = points(iv)
    if pts[2] - pts[1] < pts[1] * Coeffs.MIN_SEP || pts[end] - pts[end-1] < pts[end-1] * Coeffs.MIN_SEP
        # points closer than machine precision: neither split nor refine
        deleteat!(l.ivals, i)
    elseif iv.depth == 3 || force_split
        deleteat!(l.ivals, i)
        foreach(child -> add_ival!(l, child), split!(iv))
    else
        iv.depth += 1
        add_ival!(l, iv)
    end
    if length(l.ivals) > l.max_ivals
        weakest = l.ivals[1]
        for other in l.ivals
            (other.err, other.a) < (weakest.err, weakest.a) && (weakest = other)
        end
        deleteat!(l.ivals, findfirst(same(weakest), l.ivals))
    end
    return true
end

function ask_and_tell_pending!(l::IntegratorLearner, n)
    points, improvements = pop_from_stack!(l, n)
    while length(points) < n
        if !fill_stack!(l)
            isempty(points) && error("No way to improve the integral estimate.")
            break
        end
        new_points, new_improvements = pop_from_stack!(l, n - length(points))
        append!(points, new_points)
        append!(improvements, new_improvements)
    end
    return points, improvements
end

"""
    ask_with_improvements(learner::IntegratorLearner, n; tell_pending = true)

`n` new points and, for each, the largest error estimate of the intervals containing it (upstream's loss improvement).

With `tell_pending = false` the points are returned to the front of the queue, so asking again returns the same points
until they are told. Upstream restores a pickled state instead; that cannot work here, because the returned points
belong to intervals created by this very call and `tell!` rejects points of unknown intervals. The only state that
persists is the (idempotent) refinement of the interval that supplied the points.

**Throws**
- `ErrorException` if no interval is left to refine and no point is queued (then `done(learner)` is `true`). If some
  points are available, fewer than `n` are returned.
"""
function ask_with_improvements(l::IntegratorLearner, n::Integer; tell_pending = true)
    tell_pending && return ask_and_tell_pending!(l, n)
    xs, improvements = ask_and_tell_pending!(l, n)
    prepend!(l.stack, xs)
    return xs, improvements
end

# Intervals that currently make up the integral estimate.
approximating_intervals(l::IntegratorLearner) = l.first_ival.done_leaves

integral_estimate(l::IntegratorLearner) = sum(iv.igral for iv in approximating_intervals(l); init = 0.0)

function error_estimate(l::IntegratorLearner)
    leaves = approximating_intervals(l)
    isempty(leaves) && return Inf
    return sum(iv.err for iv in leaves)
end

"""
    npoints(learner::IntegratorLearner)

Number of evaluated (told) points.
"""
npoints(l::IntegratorLearner) = length(l.data)

"""
    result(learner::IntegratorLearner)

Current estimate as a named tuple `(integral, error, npoints)`: the integral over `bounds`, an estimate of its
absolute error (`Inf` until the first interval is complete), and the number of told points.
"""
result(l::IntegratorLearner) = (integral = integral_estimate(l), error = error_estimate(l), npoints = npoints(l))

"""
    loss(learner::IntegratorLearner; real = true)

Distance of the error estimate from the target, `abs(abs(integral) * tol - error)` as upstream. The pending points play
no role, so `real` has no effect.
"""
loss(l::IntegratorLearner; real = true) = abs(abs(integral_estimate(l)) * l.tol - error_estimate(l))

"""
    done(learner::IntegratorLearner; tol = nothing)

Whether the integral is determined. Without `tol` this is upstream's criterion: the error estimate is zero or below
`tol * abs(integral)` (the `tol` of the learner), or the part of the error that does not come from removed intervals
(those too narrow to split or at machine accuracy) is, or no interval is left. With `tol`, `loss(learner) <= tol` as for
other learners.
"""
function done(l::IntegratorLearner; tol = nothing)
    tol === nothing || return loss(l) <= tol
    err = error_estimate(l)
    igral = integral_estimate(l)
    err_excess = sum(iv.err for iv in approximating_intervals(l) if iv.removed; init = 0.0)
    return err == 0 || err < abs(igral) * l.tol || (err - err_excess < abs(igral) * l.tol < err_excess) || isempty(l.ivals)
end
