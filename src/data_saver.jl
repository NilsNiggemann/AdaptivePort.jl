# DataSaver: a wrapper that learns `arg_picker(result)` but keeps the whole `result` of every point.

"""
    DataSaver(learner; arg_picker)

Wrap `learner` for a function that returns more than the value to learn (upstream `DataSaver`). `tell!(saver, x, result)`
tells the child `arg_picker(result)` and stores `result` under the point `x`; every other interface function is
forwarded to `saver.learner`. The function `saver.f` is the child's, so runners call it and `tell!` the full result.

**Keyword Arguments**
- `arg_picker`: maps a result to the value the child learns, e.g. `r -> r.y` for a named tuple result.

**Returns**
- an [`AbstractLearner`](@ref); [`extra_data`](@ref) holds the stored results.

A repeated point overwrites its stored result, as upstream, even where the child keeps its first value
([`Learner1D`](@ref)). `to_table(saver)` is the child's table plus an `extra_data` column of the full results in row
order; saving it needs a non-`.csv` path (see [`save_data`](@ref)). `done(saver; ...)` and `has_own_goal` follow the child.

**See Also**
- [`extra_data`](@ref)
"""
struct DataSaver{L<:AbstractLearner,P,F,D<:AbstractDict} <: AbstractLearner
    learner::L
    arg_picker::P
    f::F
    extra::D
end

DataSaver(learner::AbstractLearner; arg_picker) = DataSaver(learner, arg_picker, learner.f, Dict{Any,Any}())

"""
    extra_data(saver)

The stored full results of a [`DataSaver`](@ref) as a `Dict` point => result (the live dictionary, not a copy).
Upstream uses an `OrderedDict`; here insertion order is not kept (`to_table` follows the child's row order instead).
"""
extra_data(s::DataSaver) = s.extra

tell!(s::DataSaver, x, result) = (s.extra[x] = result; tell!(s.learner, x, s.arg_picker(result)); s)
tell_pending!(s::DataSaver, x) = (tell_pending!(s.learner, x); s)
ask_with_improvements(s::DataSaver, n::Integer; tell_pending = true) = ask_with_improvements(s.learner, n; tell_pending)
loss(s::DataSaver; real = true) = loss(s.learner; real)
npoints(s::DataSaver) = npoints(s.learner)
nsamples(s::DataSaver) = nsamples(s.learner)
result(s::DataSaver) = result(s.learner)
pending_points(s::DataSaver) = pending_points(s.learner)
remove_unfinished!(s::DataSaver) = (remove_unfinished!(s.learner); s)
done(s::DataSaver; kwargs...) = done(s.learner; kwargs...)
has_own_goal(s::DataSaver) = has_own_goal(s.learner)

function to_table(s::DataSaver)
    table = to_table(s.learner)
    return merge(table, (extra_data = [s.extra[x] for (x, _) in table_rows(s.learner, table)],))
end

table_rows(s::DataSaver, table) = zip(table_rows(s.learner, table), table.extra_data)

function load_table!(s::DataSaver, table)
    for ((x, _), result) in table_rows(s, table)
        tell!(s, x, result)
    end
    return s
end

# The child's refinement path decides which points an IntegratorLearner accepts, so let it load its own table.
function load_table!(s::DataSaver{<:IntegratorLearner}, table)
    load_table!(s.learner, Base.structdiff(table, (extra_data = nothing,)))
    for ((x, _), result) in table_rows(s, table)
        haskey(s.learner.data, x) && (s.extra[x] = result)
    end
    return s
end
