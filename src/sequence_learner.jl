"""
    SequenceLearner(f; sequence, value_type = Any)

Evaluate `f(element)` for every element of `sequence`, in order. Use it to run a plain map through the runners. The points
the learner asks for are `(index, element)` with a 1-based index, and `learner.f` is the wrapper `(index, element) ->
f(element)` that runners call. Port of upstream's `SequenceLearner`.

The loss is the fraction of elements not told yet, and `done(learner)` means every element is told.

**Type Parameters**
- `V`: type of the values, from `value_type`. The default `Any` accepts everything; pass a concrete type, e.g.
  `value_type = Float64`, for typed storage and a concretely typed [`result`](@ref).

**Keyword Arguments**
- `sequence`: the elements; copied.
- `value_type`: values are converted to this type when told.

**See Also**
- [`BlockingRunner`](@ref)
"""
mutable struct SequenceLearner{F,G,S<:AbstractVector,V} <: AbstractLearner
    const f::G  # (index, element) -> userf(element), what runners call
    const userf::F
    const sequence::S
    const values::Vector{V}  # entry i is defined only if told[i]
    const told::BitVector
    const pending::BitVector
    ntold::Int
    npending::Int
    first_free::Int  # every index below this is told or pending
end

function SequenceLearner(f; sequence, value_type = Any)
    seq = collect(sequence)
    n = length(seq)
    return SequenceLearner(_ignore_index(f), f, seq, Vector{value_type}(undef, n), falses(n), falses(n), 0, 0, 1)
end

_ignore_index(f) = (_, x) -> f(x)

npoints(l::SequenceLearner) = l.ntold

function Base.show(io::IO, l::SequenceLearner)
    print(io, "SequenceLearner(", l.ntold, " of ", length(l.sequence), " told, ", l.npending, " pending)")
end

"""
    loss(learner::SequenceLearner; real = true)

Fraction of elements not told yet; with `real = false`, pending elements count as told.
"""
function loss(l::SequenceLearner; real = true)
    n = length(l.sequence)
    n == 0 && return 0.0
    return (n - l.ntold - (real ? 0 : l.npending)) / n
end

"""
    done(learner::SequenceLearner; tol = nothing)

Whether `loss(learner) <= tol`; without `tol`, whether every element is told.
"""
done(l::SequenceLearner; tol = nothing) = tol === nothing ? l.ntold == length(l.sequence) : loss(l) <= tol

has_own_goal(::SequenceLearner) = true

function ask_with_improvements(l::SequenceLearner, n::Integer; tell_pending = true)
    points = Tuple{Int,eltype(l.sequence)}[]
    i = l.first_free
    while i <= length(l.sequence) && (l.told[i] || l.pending[i])
        i += 1
    end
    l.first_free = i
    while length(points) < n && i <= length(l.sequence)
        if !l.told[i] && !l.pending[i]
            push!(points, (i, l.sequence[i]))
        end
        i += 1
    end
    tell_pending && foreach(p -> tell_pending!(l, p), points)
    return points, fill(1 / length(l.sequence), length(points))
end

"""
    tell!(learner::SequenceLearner, point, value)

Store `value` for `point = (index, element)`; only the index is used. Telling an index again overwrites its value.
"""
function tell!(l::SequenceLearner, point::Tuple{Integer,Any}, value)
    i = point[1]
    if !l.told[i]
        l.told[i] = true
        l.ntold += 1
    end
    l.pending[i] && (l.pending[i] = false; l.npending -= 1)
    l.values[i] = value
    return l
end

function tell_pending!(l::SequenceLearner, point::Tuple{Integer,Any})
    i = point[1]
    if !l.told[i] && !l.pending[i]
        l.pending[i] = true
        l.npending += 1
    end
    return l
end

pending_points(l::SequenceLearner) = [(i, l.sequence[i]) for i in findall(l.pending)]

function remove_unfinished!(l::SequenceLearner)
    fill!(l.pending, false)
    l.npending = 0
    l.first_free = 1
    return l
end

"""
    result(learner::SequenceLearner)

Named tuple `(indices, values)` of the told elements in index order. Works before the sequence is complete, where upstream
raises; once it is complete, `values` are the function values in the order of `sequence`.
"""
function result(l::SequenceLearner)
    indices = findall(l.told)
    return (indices = indices, values = l.values[indices])
end
