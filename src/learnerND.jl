# LearnerND: adaptive sampling of f: R^N -> R^M on a box, a port of `adaptive/learner/learnerND.py` from Python
# Adaptive (BSD 3-Clause, Copyright (c) 2017-2023, Adaptive authors; see the notice in triangulationND.jl).

"""
    LearnerND{N, T, V, S, F, L}

Adaptive sampler of a function `f(x1, ..., xN)` on the box `bounds`, with scalar values (`V = T`) or vector values (`V = NTuple{M, T}`), driven by [`ask`](@ref) and [`tell!`](@ref).

The evaluated points are triangulated by an own incremental Delaunay triangulation ([`TriangulationND`](@ref AdaptivePort.TriangulationND)); every simplex gets a loss, and `ask` splits the simplex with the highest loss. Pending points subdivide their simplex in a sub-triangulation whose parts share the parent's loss by volume, so repeated `ask` calls spread out.

Fields: `f` (stored for runners, never called by the learner), `bounds`, `loss_per_simplex`, `points`/`values`/`index` (evaluations in tell order and point to position), `pending`, `tri` (`nothing` until `N + 1` points not in one hyperplane exist), `losses` (simplex to loss), `subtriangulations` and `pending_to_simplex` (pending points), and `queue` (simplices sorted by priority; entries of deleted simplices are skipped when popped).

Invariants:
- `index[points[i]] == i`, `points` are distinct and inside the closed box; `pending` is disjoint from `points`.
- Vertex `i` of `tri` is `points[i]`; `losses` has one entry per simplex of `tri`.

Coordinates are scaled to the unit cube by the box widths and values by their range (all components together) before they enter a loss, as upstream. Construct with [`LearnerND(f; bounds, loss_per_simplex, vdim)`](@ref LearnerND(::Any)).

**See Also**
- [`ask`](@ref), [`tell!`](@ref), [`loss`](@ref), [`simplices`](@ref), [`default_loss_nd`](@ref)
"""
mutable struct LearnerND{N, T <: AbstractFloat, V, S <: Tuple, F, L} <: AbstractLearner
    f::F
    bounds::NTuple{N, NTuple{2, T}}
    loss_per_simplex::L
    points::Vector{NTuple{N, T}}
    values::Vector{V}
    index::Dict{NTuple{N, T}, Int}
    pending::Set{NTuple{N, T}}
    tri::Union{Nothing, TriangulationND{N, T, S}}
    losses::Dict{S, T}
    subtriangulations::Dict{S, TriangulationND{N, T, S}}
    pending_to_simplex::Dict{NTuple{N, T}, S}
    queue::Vector{Tuple{T, S, S}}
    bounds_points::Vector{NTuple{N, T}}
    widths::NTuple{N, T}
    min_value::T
    max_value::T
    old_scale::T
    output_multiplier::T
    rng::Random.Xoshiro
end

"""
    LearnerND(f; bounds = ((-1, 1), (-1, 1), (-1, 1)), loss_per_simplex = default_loss_nd, vdim = 1)

Create an empty learner for `f(x1, ..., xN)` on the box `bounds`.

**Arguments**
- `f`: the function to learn; stored but never called by the learner.

**Keyword Arguments**
- `bounds`: one `(min, max)` pair per dimension, all finite with `min < max`; the number of pairs is the dimension `N`. Integers are promoted to floating point. Upstream's convex-hull bounds are not supported.
- `loss_per_simplex`: function `(simplex, values, value_scale) -> Real`, the loss of one simplex. `simplex` is a `Vector` of `N + 1` points (`NTuple{N}`) scaled by the box widths, `values` the `Vector` of their values multiplied by `value_scale = 1 / (value range)`. See [`default_loss_nd`](@ref).
- `vdim`: number of components of the function value. `1` (default) stores plain numbers; `vdim = M > 1` stores `NTuple{M, T}`, and [`tell!`](@ref) accepts any iterable of length `M`.

**Returns**
- A [`LearnerND`](@ref); throws `ArgumentError` for invalid bounds or `vdim`.

**See Also**
- [`ask`](@ref), [`tell!`](@ref)
"""
function LearnerND(f; bounds = ((-1, 1), (-1, 1), (-1, 1)), loss_per_simplex = default_loss_nd, vdim::Integer = 1)
    b = Tuple(map(Tuple, bounds))
    N = length(b)
    N >= 1 && all(x -> length(x) == 2, b) || throw(ArgumentError("bounds must be a tuple of (min, max) pairs, got $bounds"))
    vdim >= 1 || throw(ArgumentError("vdim must be at least 1, got $vdim"))
    T = float(promote_type(map(x -> promote_type(typeof(x[1]), typeof(x[2])), b)...))
    bT = map(x -> (T(x[1]), T(x[2])), b)
    if !all(x -> isfinite(x[1]) && isfinite(x[2]) && x[1] < x[2], bT)
        throw(ArgumentError("bounds must be finite (min, max) pairs with min < max, got $bounds"))
    end
    S = NTuple{N + 1, Int}
    P = NTuple{N, T}
    V = vdim == 1 ? T : NTuple{Int(vdim), T}
    corners = sort!(vec([P(c) for c in Iterators.product(bT...)]))
    widths = map(x -> x[2] - x[1], bT)
    zero_simplex = ntuple(_ -> 0, N + 1)
    return LearnerND{N, T, V, S, typeof(f), typeof(loss_per_simplex)}(
        f, bT, loss_per_simplex, P[], V[], Dict{P, Int}(), Set{P}(), nothing, Dict{S, T}(),
        Dict{S, TriangulationND{N, T, S}}(), Dict{P, S}(), Tuple{T, S, S}[], corners, widths,
        T(Inf), T(-Inf), one(T), one(T), Random.Xoshiro(1),
    )
end

npoints(l::LearnerND) = length(l.points)

"""
    result(learner::LearnerND)

Evaluated data as a named tuple `(points, values)` of copies in tell order: `points[i]` is an `NTuple{N}` and `values[i]` its function value (a number, or an `NTuple{M}` for `vdim = M > 1`). [`simplices`](@ref) indexes into `points`.
"""
result(l::LearnerND) = (points = copy(l.points), values = copy(l.values))

pending_points(l::LearnerND) = sort!(collect(l.pending))

"""
    simplices(learner::LearnerND)

Simplices of the Delaunay triangulation of the evaluated points, as sorted index tuples into `result(learner).points`. Empty while fewer than `N + 1` points not in one hyperplane exist.

The analogue of [`triangles`](@ref) for [`Learner2D`](@ref). It is the incrementally maintained triangulation the learner samples from (Delaunay in coordinates scaled by the box widths), so it differs from a Delaunay triangulation of the raw points when the box is not a cube, and from `scipy.spatial.Delaunay` where points are cospherical.

**See Also**
- [`result`](@ref), [`LearnerND`](@ref)
"""
function simplices(l::LearnerND{N, T, V, S}) where {N, T, V, S}
    tri = get_tri!(l)
    return tri === nothing ? S[] : sort!(collect(tri.simplices))
end

# Components of a value as a tuple (a number counts as one component).
components(v::Real) = (v,)
components(v::Tuple) = v

"""
    uniform_loss_nd(simplex, values, value_scale)

Loss equal to the volume of the simplex, which samples uniformly irrespective of the values (upstream `uniform_loss`).
"""
uniform_loss_nd(simplex, values, value_scale) = volume(simplex)

"""
    std_loss_nd(simplex, values, value_scale)

Loss `r * vol^(1 / N) + vol` with `vol` the simplex volume and `r` the Euclidean norm of the per-component standard deviation (population, over the vertices) of the scaled values (upstream `std_loss`).
"""
function std_loss_nd(simplex, values, value_scale)
    vol = volume(simplex)
    M = length(components(values[1]))
    r = sqrt(sum(1:M) do c
        x = [components(v)[c] for v in values]
        sum(abs2, x .- sum(x) / length(x)) / length(x)
    end)
    return r * vol^(1 / (length(simplex) - 1)) + vol
end

"""
    default_loss_nd(simplex, values, value_scale)

Loss of upstream's `LearnerND` default: the volume of the simplex embedded in input plus output space, i.e. of the simplex with vertices `(x..., y...)`.

This is large for large simplices and for simplices on which the function changes a lot (a steep part of the graph has a large area), so it refines where the function varies without neglecting flat regions. Coordinates are scaled to the unit box and values to unit range by the learner, so the loss does not depend on the units of `f` or the box.

**Arguments**
- `simplex`: `Vector` of `N + 1` scaled points, `NTuple{N}` each.
- `values`: `Vector` of the scaled values of the vertices (numbers, or tuples for vector output).
- `value_scale`: the factor that scaled the values; unused here.

**Returns**
- The loss, a nonnegative number.

**See Also**
- [`uniform_loss_nd`](@ref), [`std_loss_nd`](@ref), [`LearnerND`](@ref)
"""
default_loss_nd(simplex, values, value_scale) = simplex_volume_in_embedding([(x..., y...) for (x, y) in zip(simplex, values)])

# The corners of the box that are neither told nor pending come first (upstream order: sorted product of the bounds).
bounds_available(l::LearnerND) = any(p -> !haskey(l.index, p) && !(p in l.pending), l.bounds_points)
bounds_done(l::LearnerND) = all(p -> haskey(l.index, p), l.bounds_points)

scale_point(l::LearnerND{N}, p) where {N} = ntuple(i -> p[i] / l.widths[i], N)

function to_point(l::LearnerND{N, T}, p) where {N, T}
    length(p) == N || throw(ArgumentError("point $p has $(length(p)) coordinates, expected $N"))
    q = ntuple(i -> T(p[i]), N)
    all(i -> l.bounds[i][1] <= q[i] <= l.bounds[i][2], 1:N) || throw(ArgumentError("point $p is outside the bounds $(l.bounds)"))
    return q
end

to_value(::LearnerND{N, T, T}, y) where {N, T} = y isa Real ? T(y) : throw(ArgumentError("expected a number as value (vdim = 1), got $y"))
function to_value(::LearnerND{N, T, NTuple{M, T}}, y) where {N, T, M}
    v = collect(y)
    length(v) == M || throw(ArgumentError("expected a value with $M components (vdim = $M), got $(length(v))"))
    return ntuple(i -> T(v[i]), M)
end

# The triangulation of all evaluated points, built on first need (upstream's lazy `tri` property); `nothing` while impossible.
function get_tri!(l::LearnerND)
    l.tri === nothing || return l.tri
    length(l.points) > length(l.bounds) || return nothing
    tri = try
        TriangulationND(l.points)
    catch err
        err isa ArgumentError || rethrow()
        return nothing  # points in one hyperplane: more points are needed
    end
    l.tri = tri
    update_losses!(l, empty(tri.simplices), tri.simplices)
    return tri
end

function compute_loss(l::LearnerND{N, T}, simplex) where {N, T}
    pts = [scale_point(l, l.points[i]) for i in simplex]
    vals = [l.output_multiplier .* l.values[i] for i in simplex]
    return T(l.loss_per_simplex(pts, vals, l.output_multiplier))
end

# Insert into the sorted queue. Priority as upstream: highest loss first (rounded to 8 digits so that equal losses tie
# deterministically), then simplex, then subsimplex (all zero for a real simplex, which sorts before any subsimplex).
# ponytail: sorted Vector, O(n) insertion and O(1) pop; use a heap with lazy deletion if the queue passes ~1e5 entries.
function queue_add!(l::LearnerND{N, T, V, S}, loss, simplex, sub = ntuple(_ -> 0, N + 1)) where {N, T, V, S}
    key = (T(-round(loss; digits = 8)), simplex, sub)
    insert!(l.queue, searchsortedfirst(l.queue, key), key)
    return nothing
end

function update_subsimplex_losses!(l::LearnerND, simplex, new_subsimplices)
    density = l.losses[simplex] / volume(l.tri, simplex)
    st = l.subtriangulations[simplex]
    for sub in sort!(collect(new_subsimplices))
        queue_add!(l, volume(st, sub) * density, simplex, sub)
    end
end

function add_pending_point_to_simplex!(l::LearnerND, point, simplex)
    st = get!(() -> TriangulationND(l.tri.vertices[collect(simplex)]), l.subtriangulations, simplex)
    l.pending_to_simplex[point] = simplex
    return add_point!(st, point)
end

function try_adding_pending_point_to_simplex!(l::LearnerND, point, simplex)
    point_in_simplex(l.tri, point, simplex) && add_pending_point_to_simplex!(l, point, simplex)
    return nothing
end

# Losses for the new simplices `to_add`, dropping those of `to_delete`; pending points of dropped simplices move to the new ones.
function update_losses!(l::LearnerND{N, T}, to_delete, to_add) where {N, T}
    unbound = Set{NTuple{N, T}}()
    for s in to_delete
        delete!(l.losses, s)
        st = pop!(l.subtriangulations, s, nothing)
        st === nothing || union!(unbound, st.vertices)
    end
    filter!(p -> !haskey(l.index, p), unbound)
    for s in sort!(collect(to_add))
        l.losses[s] = compute_loss(l, s)
        for p in sort!(collect(unbound))
            try_adding_pending_point_to_simplex!(l, p, s)
        end
        if haskey(l.subtriangulations, s)
            update_subsimplex_losses!(l, s, l.subtriangulations[s].simplices)
        else
            queue_add!(l, l.losses[s], s)
        end
    end
    return nothing
end

# Rebuild the queue from the stored simplex losses (after the output scale changed or pending points were dropped).
function rebuild_queue!(l::LearnerND)
    empty!(l.queue)
    for s in sort!(collect(keys(l.losses)))
        if haskey(l.subtriangulations, s)
            update_subsimplex_losses!(l, s, l.subtriangulations[s].simplices)
        else
            queue_add!(l, l.losses[s], s)
        end
    end
    return nothing
end

function recompute_all_losses!(l::LearnerND)
    l.tri === nothing && return nothing
    for s in l.tri.simplices
        l.losses[s] = compute_loss(l, s)
    end
    return rebuild_queue!(l)
end

# Track the value range and rescale all losses once the range has grown by more than 10 % (upstream's `_update_range`).
function update_range!(l::LearnerND{N, T}, value) where {N, T}
    lo, hi = extrema(components(value))
    if isinf(l.min_value)
        l.min_value, l.max_value = lo, hi
        l.old_scale = l.max_value > l.min_value ? l.max_value - l.min_value : one(T)
        return nothing
    end
    l.min_value, l.max_value = min(l.min_value, lo), max(l.max_value, hi)
    scale = l.max_value - l.min_value
    multiplier = 1 / (scale > 0 ? scale : one(T))
    # do not rescale if the rounding error of the scaled values would exceed 1e-2
    1e-15 * max(abs(l.min_value), abs(l.max_value)) * multiplier > 1e-2 && (multiplier = one(T))
    l.output_multiplier = multiplier
    if scale / l.old_scale > 1.1
        l.old_scale = scale
        recompute_all_losses!(l)
    end
    return nothing
end

"""
    tell!(learner::LearnerND, x, y)

Record the evaluation `y = f(x...)` at the point `x` (a tuple or vector of `N` coordinates); returns `learner`.

`y` is a number, or for `vdim = M > 1` any iterable of `M` numbers. Throws `ArgumentError` if `x` is outside the bounds (deviation: upstream stores but ignores such points), if `y` has the wrong length, or if `y` is not finite (deviation: upstream does not check). Telling an evaluated point again is ignored, as upstream. The triangulation and the losses of the affected simplices are updated incrementally.
"""
function tell!(l::LearnerND{N, T, V}, x, y) where {N, T, V}
    p = to_point(l, x)
    v = to_value(l, y)
    all(isfinite, components(v)) || throw(ArgumentError("value $y is not finite"))
    haskey(l.index, p) && return l
    delete!(l.pending, p)
    tri = get_tri!(l)
    push!(l.points, p)
    push!(l.values, v)
    l.index[p] = length(l.points)
    update_range!(l, v)
    if tri !== nothing
        simplex = pop!(l.pending_to_simplex, p, nothing)
        simplex in tri.simplices || (simplex = nothing)
        # the Delaunay criterion is evaluated in the scaled coordinates
        to_delete, to_add = add_point!(tri, p; simplex, transform = scale_transform(l))
        update_losses!(l, to_delete, to_add)
    end
    return l
end

scale_transform(l::LearnerND{N, T}) where {N, T} = Matrix(LinearAlgebra.Diagonal(collect(1 ./ l.widths)))

# All simplices that contain `point`, starting the search from the simplex `simplex` if given.
function simplices_containing(l::LearnerND, point, simplex)
    simplex === nothing && (simplex = locate_point(l.tri, point))
    simplex === nothing && return empty(l.tri.simplices)
    candidates = union((l.tri.vertex_to_simplices[v] for v in simplex)...)
    return sort!([s for s in candidates if point_in_simplex(l.tri, point, s)])
end

"""
    tell_pending!(learner::LearnerND, x)

Mark the point `x` as issued but not evaluated; returns `learner`. If a triangulation exists, `x` subdivides the simplices that contain it in sub-triangulations that share the loss of the parent simplex by volume, so later `ask` calls avoid its neighbourhood.

Throws `ArgumentError` if `x` is outside the bounds (deviation: upstream ignores such points). Points that are already evaluated are ignored.
"""
function tell_pending!(l::LearnerND, x; simplex = nothing)
    p = to_point(l, x)
    haskey(l.index, p) && return l
    push!(l.pending, p)
    get_tri!(l) === nothing && return l
    for s in simplices_containing(l, p, simplex)
        _, added = add_pending_point_to_simplex!(l, p, s)
        update_subsimplex_losses!(l, s, added)
    end
    return l
end

"""
    remove_unfinished!(learner::LearnerND)

Forget all pending points and their sub-triangulations; returns `learner`. The simplex queue is rebuilt from the stored losses (deviation: upstream leaves the queue as it is and so can lose simplices that were subdivided by pending points).
"""
function remove_unfinished!(l::LearnerND)
    empty!(l.pending)
    empty!(l.subtriangulations)
    empty!(l.pending_to_simplex)
    rebuild_queue!(l)
    return l
end

"""
    loss(learner::LearnerND; real = true)

Largest simplex loss, `Inf` until the box corners are evaluated and a triangulation exists (deviation: upstream returns the loss of the first triangulation, however incomplete).

With `real = false` pending points count: the loss of a simplex that is subdivided by pending points is replaced by that of its largest part. Upstream's `loss(real = False)` is not implemented and returns the same as `real = true`.
"""
function loss(l::LearnerND{N, T}; real::Bool = true) where {N, T}
    bounds_done(l) || return T(Inf)
    tri = get_tri!(l)
    (tri === nothing || isempty(l.losses)) && return T(Inf)
    real && return maximum(values(l.losses))
    return maximum(l.losses) do (s, ls)
        st = get(l.subtriangulations, s, nothing)
        st === nothing ? ls : ls * maximum(volumes(st)) / volume(tri, s)
    end
end

"""
    choose_point_in_simplex(simplex, widths)

New point inside the simplex (`N + 1` points of `NTuple{N}`): its centre if the circumcentre lies inside it in the box-scaled coordinates (`x / widths`), otherwise the midpoint of its longest edge there (upstream `choose_point_in_simplex`). The result is in the original coordinates.
"""
function choose_point_in_simplex(simplex, widths)
    N = length(widths)
    s = [ntuple(i -> p[i] / widths[i], N) for p in simplex]
    center = try
        first(circumsphere(s))
    catch err
        err isa LinearAlgebra.SingularException || rethrow()
        nothing
    end
    if center !== nothing && point_in_simplex(center, s)
        point = ntuple(k -> sum(p[k] for p in s) / length(s), N)
    else
        best, (i, j) = -Inf, (1, 2)
        for a in eachindex(s), b in (a + 1):length(s)
            d = sum(abs2, s[a] .- s[b])
            d > best && ((best, i, j) = (d, a, b))
        end
        point = (s[i] .+ s[j]) ./ 2
    end
    return ntuple(k -> point[k] * widths[k], N)
end

# Pop the highest-priority entry that still exists (entries of deleted or subdivided simplices are stale).
function pop_highest_existing_simplex!(l::LearnerND)
    zero_simplex = ntuple(_ -> 0, length(l.bounds) + 1)
    while !isempty(l.queue)
        negloss, simplex, sub = popfirst!(l.queue)
        simplex in l.tri.simplices || continue
        if sub == zero_simplex
            haskey(l.subtriangulations, simplex) || return -negloss, simplex, nothing
        elseif haskey(l.subtriangulations, simplex) && sub in l.subtriangulations[simplex].simplices
            return -negloss, simplex, sub
        end
    end
    error("no simplex left to subdivide in a non-empty triangulation")
end

function ask_best_point!(l::LearnerND{N, T}) where {N, T}
    while true
        loss, simplex, sub = pop_highest_existing_simplex!(l)
        verts = sub === nothing ? l.tri.vertices[collect(simplex)] : l.subtriangulations[simplex].vertices[collect(sub)]
        p = choose_point_in_simplex(verts, l.widths)
        p = ntuple(i -> clamp(p[i], l.bounds[i]...), N)  # rounding of the scaling can leave the box by an ulp
        # stale queue entries from before a pending split may propose a known point: skip them
        (haskey(l.index, p) || p in l.pending) && continue
        l.pending_to_simplex[p] = simplex
        tell_pending!(l, p; simplex)
        return p, T(loss)
    end
end

function ask_one!(l::LearnerND{N, T}) where {N, T}
    if bounds_available(l)
        p = first(c for c in l.bounds_points if !haskey(l.index, c) && !(c in l.pending))
        tell_pending!(l, p)
        return p, T(Inf)
    end
    if get_tri!(l) === nothing  # no triangulation yet: a random point in the box
        p = ntuple(i -> l.bounds[i][1] + rand(l.rng, T) * l.widths[i], N)
        tell_pending!(l, p)
        return p, T(Inf)
    end
    return ask_best_point!(l)
end

# Run `body()` and restore every field that `ask` mutates afterwards.
# ponytail: copies the queue and all sub-triangulations per call, O(queue + pending simplices); record touched
# entries and undo them if this shows up in profiles (the runner asks with tell_pending = false every iteration).
function with_restore(body, l::LearnerND)
    get_tri!(l)  # a lazily created triangulation fills the queue and must not be rolled back
    pending, queue, p2s = copy(l.pending), copy(l.queue), copy(l.pending_to_simplex)
    subtris = Dict(k => copy(v) for (k, v) in l.subtriangulations)
    try
        return body()
    finally
        l.pending, l.queue, l.pending_to_simplex, l.subtriangulations = pending, queue, p2s, subtris
    end
end

"""
    ask_with_improvements(learner::LearnerND, n; tell_pending = true)

Propose up to `n` new points `NTuple{N}` to evaluate, with their expected loss improvements (the loss of the simplex each point splits, `Inf` for the corners and random points), as `(points, improvements)`. [`ask`](@ref) returns the points only.

Upstream's order: first the corners of the box that are neither evaluated nor pending (in lexicographic order); a seeded random point while no triangulation exists yet; afterwards the simplex (or pending sub-simplex) with the highest loss is split at its centre, or at the midpoint of its longest edge if the circumcentre lies outside it. The returned points are never evaluated or pending.

**Keyword Arguments**
- `tell_pending`: register the points via [`tell_pending!`](@ref) (default), so that repeated calls never repeat a point. With `false` the learner is restored after the call (upstream does the same), at the cost of copying its pending data.

**See Also**
- [`LearnerND`](@ref), [`tell!`](@ref)
"""
function ask_with_improvements(l::LearnerND{N, T}, n::Integer; tell_pending::Bool = true) where {N, T}
    n < 0 && throw(ArgumentError("n must be non-negative, got $n"))
    function body()
        points, losses = NTuple{N, T}[], T[]
        for _ in 1:n
            p, ls = ask_one!(l)
            push!(points, p)
            push!(losses, ls)
        end
        return points, losses
    end
    return tell_pending ? body() : with_restore(body, l)
end
