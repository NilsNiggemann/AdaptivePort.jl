"""
    Dispatch(fs)

Callable `(index, point) -> fs[index](point...)` that routes a point of a [`BalancingLearner`](@ref) to the function
of its child. It holds the functions only, never the learners, so a runner that ships it to remote workers does not
ship the learners' data. The point is splatted because points of 2D and ND learners are tuples.
"""
struct Dispatch{F<:AbstractVector}
    fs::F
end

(d::Dispatch)(i, x) = d.fs[i](x...)

const BALANCING_STRATEGIES = (:loss_improvements, :loss, :npoints, :cycle)

"""
    BalancingLearner(learners; strategy = :loss_improvements, cdims = nothing)

Choose the points to evaluate from several learners of the same kind, so that a single runner serves all of them
(port of upstream's `BalancingLearner`). The points are `(index, point)` with a 1-based `index` of the child
(deviation from upstream's 0-based index), and `learner.f` is `f(index, point)`, which calls the child's function.
Children stay independently usable, but telling them directly bypasses the balancer's caches; use
`tell!(balancer, (i, x), y)`.

Their losses are compared, so all children must be the same kind (the same unparameterised type) and the losses
must be comparable; if they are not, choose another `strategy`.

**Keyword Arguments**
- `strategy`: `:loss_improvements` (default) picks the child whose next point promises the largest loss
  improvement, `:loss` the child with the largest loss (counting pending points), `:npoints` the child with the
  fewest points, `:cycle` goes through the children in turn. Ties go to the child with the fewest evaluated plus
  pending points. Children that are finished (`done(child)` for learners with an own goal) are skipped. The
  strategy is validated when set (`balancer.strategy = :loss`) and may be changed while running.
- `cdims`: vector of named tuples labelling the children (default `(i = 1,)`, `(i = 2,)`, ...).

`loss` is the largest child loss, `npoints` the sum, `result` the vector of child results. `done(balancer)` without
`tol` is available (and the runner stops on it) when the children have an own goal, and means that all are done.
`ask(...; tell_pending = false)` leaves the children untouched (the points chosen meanwhile are made un-pending
again, and the `:cycle` position is restored too). Exception: for an [`IntegratorLearner`](@ref) child, pending
points that existed before the call are re-queued instead of kept pending, and the queue order may change.

Deviation: upstream invalidates only the real loss cache on `tell_pending`; here both cached losses are.

**See Also**
- [`AdaptivePort.from_product`](@ref)
"""
mutable struct BalancingLearner{L<:AbstractLearner,D<:Dispatch,C<:AbstractVector} <: AbstractLearner
    const learners::Vector{L}
    const f::D
    const cdims::C
    strategy::Symbol
    cycle_pos::Int
    const ask_cache::Dict{Int,Any}  # child index => (points, improvements) from ask(child, 1; tell_pending = false)
    const loss_cache::Dict{Int,Float64}
    const pending_loss_cache::Dict{Int,Float64}
end

function BalancingLearner(learners; strategy::Symbol = :loss_improvements, cdims = nothing)
    ls = collect(learners)
    isempty(ls) && throw(ArgumentError("a BalancingLearner needs at least one learner"))
    all(l -> l isa AbstractLearner, ls) || throw(ArgumentError("all children must be learners"))
    kind(l) = Base.typename(typeof(l)).wrapper
    all(l -> kind(l) === kind(first(ls)), ls) || throw(ArgumentError("a BalancingLearner can handle only one kind of learner, got $(unique(kind.(ls)))"))
    check_strategy(strategy)
    labels = cdims === nothing ? [(i = i,) for i in eachindex(ls)] : collect(cdims)
    length(labels) == length(ls) || throw(ArgumentError("cdims has $(length(labels)) entries for $(length(ls)) learners"))
    return BalancingLearner(ls, Dispatch([l.f for l in ls]), labels, strategy, 1, Dict{Int,Any}(), Dict{Int,Float64}(), Dict{Int,Float64}())
end

function check_strategy(strategy)
    strategy in BALANCING_STRATEGIES || throw(ArgumentError("strategy must be one of $BALANCING_STRATEGIES, got :$strategy"))
    return strategy
end

function Base.setproperty!(b::BalancingLearner, name::Symbol, value)
    name === :strategy && check_strategy(value)
    return setfield!(b, name, convert(fieldtype(typeof(b), name), value))
end

function Base.show(io::IO, b::BalancingLearner)
    print(io, "BalancingLearner(", length(b.learners), " ", Base.typename(typeof(first(b.learners))).name, ", strategy = :", b.strategy, ")")
end

"""
    from_product(f, learner_type, learner_kwargs, combos; strategy = :loss_improvements)

[`BalancingLearner`](@ref) with one child `learner_type((args...) -> f(args...; combo...); learner_kwargs...)` per
combination of the named parameter values in `combos` (a `NamedTuple` of collections, or a `Vector` of `name => values`
pairs). The first parameter varies slowest, as upstream's `named_product`. `cdims` holds the combinations as named
tuples. Unexported; call as `AdaptivePort.from_product`.

```julia
AdaptivePort.from_product((x; a, b) -> a * x + b, Learner1D, (bounds = (-1, 1),), (a = 1:3, b = [0.0, 1.0]))
```
"""
function from_product(f, learner_type, learner_kwargs, combos; strategy = :loss_improvements)
    names = Tuple(Symbol(first(p)) for p in pairs_of(combos))
    values = [last(p) for p in pairs_of(combos)]
    cdims = vec([NamedTuple{names}(Tuple(reverse(c))) for c in Iterators.product(reverse(values)...)])
    learners = [learner_type((args...) -> f(args...; pairs(combo)...); learner_kwargs...) for combo in cdims]
    return BalancingLearner(learners; cdims, strategy)
end

pairs_of(combos::NamedTuple) = collect(pairs(combos))
pairs_of(combos::AbstractVector) = combos

# Marks `x` as handed out after an `ask(child, 1; tell_pending = false)`. The integrator tracks its own pending points
# by keeping them out of its queue, so committing means dequeuing.
mark_pending!(l::AbstractLearner, x) = tell_pending!(l, x)
function mark_pending!(l::IntegratorLearner, x)
    i = findfirst(==(x), l.stack)
    i === nothing || deleteat!(l.stack, i)
    return l
end
mark_pending!(s::DataSaver, x) = (mark_pending!(s.learner, x); s)

finished(l::AbstractLearner) = has_own_goal(l) && done(l)

total_points(l::AbstractLearner) = npoints(l) + length(pending_points(l))

function invalidate!(b::BalancingLearner, i)
    delete!(b.ask_cache, i)
    delete!(b.loss_cache, i)
    delete!(b.pending_loss_cache, i)
    return b
end

function clear_caches!(b::BalancingLearner)
    empty!(b.ask_cache)
    empty!(b.loss_cache)
    empty!(b.pending_loss_cache)
    return b
end

function child_loss(b::BalancingLearner, i, real)
    cache = real ? b.loss_cache : b.pending_loss_cache
    return get!(() -> Float64(loss(b.learners[i]; real)), cache, i)
end

function cached_ask(b::BalancingLearner, i)
    return get!(() -> ask_with_improvements(b.learners[i], 1; tell_pending = false), b.ask_cache, i)
end

# Each method returns `(child index, point, improvement)` and marks the point pending, or `nothing` if no child can give one.
function next_point!(b::BalancingLearner, total, ::Val{:loss_improvements}, skip)
    best = nothing
    for i in eachindex(b.learners)
        (i in skip || finished(b.learners[i])) && continue
        points, improvements = cached_ask(b, i)
        isempty(points) && continue
        score = (improvements[1], -total[i])
        (best === nothing || score > best[2]) && (best = (i, score))
    end
    best === nothing && return nothing
    i = best[1]
    points, improvements = cached_ask(b, i)
    mark_pending!(b.learners[i], points[1])
    return i, points[1], improvements[1]
end

function next_point!(b::BalancingLearner, total, ::Val{:loss}, skip)
    best = nothing
    for i in eachindex(b.learners)
        (i in skip || finished(b.learners[i])) && continue
        score = (child_loss(b, i, false), -total[i])
        (best === nothing || score > best[2]) && (best = (i, score))
    end
    return best === nothing ? nothing : ask_child!(b, best[1], total, skip)
end

function next_point!(b::BalancingLearner, total, ::Val{:npoints}, skip)
    best = nothing
    for i in eachindex(b.learners)
        (i in skip || finished(b.learners[i])) && continue
        (best === nothing || total[i] < total[best]) && (best = i)
    end
    return best === nothing ? nothing : ask_child!(b, best, total, skip)
end

function next_point!(b::BalancingLearner, total, ::Val{:cycle}, skip)
    n = length(b.learners)
    for _ in 1:n
        i = b.cycle_pos
        b.cycle_pos = mod1(i + 1, n)
        (i in skip || finished(b.learners[i])) && continue
        return ask_child!(b, i, total, skip)
    end
    return nothing
end

# One pending point from child `i`; a child with none left is added to `skip` and the choice is made again.
function ask_child!(b::BalancingLearner, i, total, skip)
    points, improvements = ask_with_improvements(b.learners[i], 1)
    if isempty(points)
        push!(skip, i)
        return next_point!(b, total, Val(b.strategy), skip)
    end
    return i, points[1], improvements[1]
end

function ask_and_tell_pending!(b::BalancingLearner, n)
    total = [total_points(l) for l in b.learners]
    chosen = Any[]
    improvements = Float64[]
    skip = Set{Int}()
    strategy = Val(b.strategy)
    while length(chosen) < n
        picked = next_point!(b, total, strategy, skip)
        picked === nothing && break
        i, point, improvement = picked
        total[i] += 1
        invalidate!(b, i)
        push!(chosen, (i, point))
        push!(improvements, improvement)
    end
    return map(identity, chosen), improvements
end

"""
    ask_with_improvements(balancer::BalancingLearner, n; tell_pending = true)

Up to `n` points `(index, point)` with the loss improvement of each, chosen by `balancer.strategy`. Fewer (or no)
points are returned when the children have none left to give (all finished). With `tell_pending = false` the children
and the `:cycle` position are restored afterwards.
"""
function ask_with_improvements(b::BalancingLearner, n::Integer; tell_pending = true)
    n == 0 && return Tuple{Int,Any}[], Float64[]
    tell_pending && return ask_and_tell_pending!(b, n)
    snapshots = [pending_points(l) for l in b.learners]
    cycle_pos = b.cycle_pos
    try
        return ask_and_tell_pending!(b, n)
    finally
        for (l, pending) in zip(b.learners, snapshots)
            remove_unfinished!(l)
            foreach(p -> mark_pending!(l, p), pending)
        end
        b.cycle_pos = cycle_pos
        clear_caches!(b)
    end
end

"""
    tell!(balancer::BalancingLearner, point, value)

Route `value` to child `point[1]` for its point `point[2]`. Returns `balancer`.
"""
function tell!(b::BalancingLearner, point::Tuple{Integer,Any}, value)
    i, x = point
    tell!(b.learners[i], x, value)
    invalidate!(b, i)
    return b
end

"""
    tell_pending!(balancer::BalancingLearner, point)

Mark `point = (index, x)` as pending in child `index`. Returns `balancer`.
"""
function tell_pending!(b::BalancingLearner, point::Tuple{Integer,Any})
    i, x = point
    tell_pending!(b.learners[i], x)
    invalidate!(b, i)
    return b
end

"""
    loss(balancer::BalancingLearner; real = true)

Largest child loss; child losses are cached until the child is told or asked through the balancer.
"""
loss(b::BalancingLearner; real = true) = maximum(i -> child_loss(b, i, real), eachindex(b.learners))

"""
    done(balancer::BalancingLearner; tol = nothing)

With `tol`, whether `loss(balancer) <= tol`. Without, whether every child is done (needs children with an own goal).

**Throws**
- `ArgumentError` for `tol = nothing` if the children have no own goal.
"""
function done(b::BalancingLearner; tol = nothing)
    tol === nothing || return loss(b) <= tol
    has_own_goal(b) || throw(ArgumentError("done(balancer) without `tol` needs children with an own goal"))
    return all(done, b.learners)
end

has_own_goal(b::BalancingLearner) = all(has_own_goal, b.learners)

"""
    npoints(balancer::BalancingLearner)

Sum of the children's number of points.
"""
npoints(b::BalancingLearner) = sum(npoints, b.learners)

nsamples(b::BalancingLearner) = sum(nsamples, b.learners)

"""
    result(balancer::BalancingLearner)

Vector of the children's results.
"""
result(b::BalancingLearner) = [result(l) for l in b.learners]

"""
    pending_points(balancer::BalancingLearner)

Pending points of all children as `(index, point)`.
"""
pending_points(b::BalancingLearner) = [(i, p) for (i, l) in enumerate(b.learners) for p in pending_points(l)]

"""
    remove_unfinished!(balancer::BalancingLearner)

Forget the pending points of all children. Returns `balancer`.
"""
function remove_unfinished!(b::BalancingLearner)
    foreach(remove_unfinished!, b.learners)
    clear_caches!(b)
    return b
end

"""
    to_table(balancer::BalancingLearner)

The tables of the children stacked, with the child index (1-based) as first column `learner_index`.
"""
function to_table(b::BalancingLearner)
    tables = [to_table(l) for l in b.learners]
    index = reduce(vcat, [fill(i, length(first(t))) for (i, t) in enumerate(tables)])
    return merge((learner_index = index,), map((cols...) -> reduce(vcat, cols), tables...))
end

# Rows of `table` that belong to child `i`, without the index column.
function child_table(table, i)
    mask = table.learner_index .== i
    return map(col -> col[mask], Base.structdiff(table, NamedTuple{(:learner_index,)}))
end

function table_rows(b::BalancingLearner, table)
    rows = Any[]
    for (i, l) in enumerate(b.learners)
        append!(rows, (((i, x), y) for (x, y) in table_rows(l, child_table(table, i))))
    end
    return rows
end

function load_table!(b::BalancingLearner, table)
    all(i -> 1 <= i <= length(b.learners), table.learner_index) || throw(ArgumentError("table has learner_index outside 1:$(length(b.learners))"))
    for (i, l) in enumerate(b.learners)
        sub = child_table(table, i)
        isempty(first(sub)) || load_table!(l, sub)
    end
    clear_caches!(b)
    return b
end
