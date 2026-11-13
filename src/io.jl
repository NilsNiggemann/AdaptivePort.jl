# Data export (`to_table`) and saving (`save_data`, `load_data!`). Per-learner methods live here so that the
# learner files stay unaware of file formats.

"""
    to_table(learner)

The told evaluations of `learner` as a named tuple of equal-length vectors, one row per evaluation (upstream
`to_dataframe`). It is a Tables.jl column table, so `DataFrame(to_table(l))` works without this package depending on
DataFrames. Never mutates `learner`; pending points are not included.

**Columns**
- [`Learner1D`](@ref): `x, y` (sorted by `x`); vector output `x, y_1, ..., y_M`.
- [`Learner2D`](@ref): `x, y, value` in tell order; vector output `value_1, ..., value_M`.
- [`LearnerND`](@ref): `x, y, z` for `N <= 3`, else `x_1, ..., x_N`; then `value` or `value_1, ...`; tell order.
- [`AverageLearner`](@ref): `seed, y`. [`AverageLearner1D`](@ref): `seed, x, y`, one row per sample.
- [`IntegratorLearner`](@ref): `x, y`. [`SequenceLearner`](@ref): `index, x, y` (1-based `index`; upstream's `i` is 0-based).

**See Also**
- [`save_data`](@ref), [`load_data!`](@ref)
"""
function to_table end

"""
    table_rows(learner, table)

Iterator of `(point, value)` pairs of a table with the columns of `to_table(learner)`, such that `tell!(learner, point,
value)` replays it. Used by [`load_table!`](@ref).
"""
function table_rows end

"""
    load_table!(learner, table)

Replay the rows of `table` (columns as in [`to_table`](@ref)) into `learner` with `tell!`; returns `learner`. The generic
method suffices where `tell!` accepts arbitrary data; learners that cannot replay arbitrary data override it.
"""
function load_table!(l::AbstractLearner, table)
    for (point, value) in table_rows(l, table)
        tell!(l, point, value)
    end
    return l
end

"""
    save_data(path, learner)

Write the told data of `learner` to `path`; never the function, the learner struct or tasks. The extension picks the
format: `.csv` is a header plus one row per point of `to_table(learner)` (numbers only, floats in shortest round-trip
form so nothing is lost, readable by pandas and NumPy); any other extension (suggest `.jls`) is a `Serialization` file
holding a named tuple with a format tag, a version number and the table.

The file is written to a temporary file next to `path` and then moved over it, so a reader or a crash never sees a
half-written file.

**Limits**
- Pending points are not saved.
- `Serialization` files are tied to the Julia version, and loading one runs `deserialize`: only load your own files.
- After [`load_data!`](@ref) the data are identical, but later sampling decisions can differ from an uninterrupted run
  where a learner's state depends on history (the lazy y-scale of [`Learner1D`](@ref)), as upstream.

**Throws**
- `ArgumentError` for a `.csv` path if a column holds non-numbers; use a `.jls` path (for a `DataSaver`, which has a
  non-numeric `extra_data` column, or save `saver.learner`).

**See Also**
- [`load_data!`](@ref), [`to_table`](@ref)
"""
function save_data(path::AbstractString, l::AbstractLearner)
    table = to_table(l)
    if is_csv(path)
        text = csv_text(table)
        write_atomically(io -> write(io, text), path)
    else
        write_atomically(io -> Serialization.serialize(io, (format = FORMAT_TAG, version = FORMAT_VERSION, table = table)), path)
    end
    return path
end

"""
    load_data!(learner, path)

Fill a freshly constructed `learner` (the function is supplied by the constructor) with the data of a file written by
[`save_data`](@ref), by replaying `tell!`; returns `learner`. The format follows the extension as in `save_data`. To
resume a run, use `isfile(path) && load_data!(learner, path)`. The limits of `save_data` apply.

**Throws**
- `ArgumentError` if the file does not exist, is not a file of this package, or its columns differ from those the
  learner expects (the message lists both).

**See Also**
- [`save_data`](@ref), [`load_table!`](@ref)
"""
function load_data!(l::AbstractLearner, path::AbstractString)
    isfile(path) || throw(ArgumentError("data file not found: $path"))
    table = is_csv(path) ? read_csv(path) : read_jls(path)
    expected = keys(to_table(l))
    found = keys(table)
    expected == found || throw(ArgumentError("columns of $path do not match the learner: expected $(collect(expected)), found $(collect(found))"))
    return load_table!(l, table)
end

const FORMAT_TAG = "AdaptivePort.data"
const FORMAT_VERSION = 1

is_csv(path) = lowercase(splitext(path)[2]) == ".csv"

function write_atomically(writer, path)
    tmp, io = mktemp(dirname(abspath(path)))
    try
        writer(io)
        close(io)
        mv(tmp, path; force = true)
    catch
        close(io)
        rm(tmp; force = true)
        rethrow()
    end
    return path
end

function read_jls(path)
    data = Serialization.deserialize(path)
    (data isa NamedTuple && get(data, :format, nothing) == FORMAT_TAG) || throw(ArgumentError("$path is not an AdaptivePort data file"))
    data.version == FORMAT_VERSION || throw(ArgumentError("$path has format version $(data.version), expected $FORMAT_VERSION"))
    return data.table
end

# Floats print in shortest round-trip form; `NaN`, `Inf` and `-Inf` are read back by pandas as floats.
function csv_text(table)
    for (name, col) in pairs(table)
        all(x -> x isa Real, col) || throw(ArgumentError("CSV holds numbers only, but column `$name` does not; use a `.jls` path instead (for a `DataSaver`, also possible: save `saver.learner`)"))
    end
    io = IOBuffer()
    println(io, join(keys(table), ","))
    for i in 1:length(first(table))
        println(io, join((col[i] for col in table), ","))
    end
    return String(take!(io))
end

function read_csv(path)
    lines = readlines(path)
    isempty(lines) && throw(ArgumentError("$path is empty"))
    names = Symbol.(split(lines[1], ","))
    cells = [split(line, ",") for line in lines[2:end] if !isempty(line)]
    all(row -> length(row) == length(names), cells) || throw(ArgumentError("$path has rows with a different number of fields than its header"))
    columns = map(eachindex(names)) do j
        strs = [row[j] for row in cells]
        ints = tryparse.(Int, strs)
        any(isnothing, ints) || return Vector{Int}(ints)
        floats = tryparse.(Float64, strs)
        any(isnothing, floats) && throw(ArgumentError("$path: column `$(names[j])` holds non-numbers"))
        return Vector{Float64}(floats)
    end
    return NamedTuple{Tuple(names)}(Tuple(columns))
end

# `(name = col)` or, for tuple values, `(name_1 = ..., name_2 = ...)`.
value_columns(name::Symbol, values::Vector{<:Real}) = NamedTuple{(name,)}((copy(values),))
function value_columns(name::Symbol, values::Vector{V}) where {V<:Tuple}
    M = fieldcount(V)
    return NamedTuple{ntuple(j -> Symbol(name, :_, j), M)}(ntuple(j -> [v[j] for v in values], M))
end

# The values of a table as one iterable: the column `name`, or tuples of the columns `name_1`, `name_2`, ...
function values_of(table, name::Symbol)
    haskey(table, name) && return table[name]
    prefix = string(name, "_")
    return zip((table[k] for k in keys(table) if startswith(string(k), prefix))...)
end

# The first `n` columns as an iterable of tuples.
points_of(table, n) = zip(values(table)[1:n]...)

# Learner1D
to_table(l::Learner1D) = merge((x = copy(l.xs),), value_columns(:y, l.ys))
table_rows(::Learner1D, table) = zip(table.x, values_of(table, :y))

# Learner2D
function to_table(l::Learner2D)
    return merge((x = [p[1] for p in l.points], y = [p[2] for p in l.points]), value_columns(:value, l.values))
end
table_rows(::Learner2D, table) = zip(points_of(table, 2), values_of(table, :value))

# LearnerND
point_names(N) = N <= 3 ? (:x, :y, :z)[1:N] : ntuple(j -> Symbol(:x_, j), N)
function to_table(l::LearnerND{N}) where {N}
    coords = NamedTuple{point_names(N)}(ntuple(j -> [p[j] for p in l.points], N))
    return merge(coords, value_columns(:value, l.values))
end
table_rows(::LearnerND{N}, table) where {N} = zip(points_of(table, N), values_of(table, :value))

# AverageLearner
function to_table(l::AverageLearner)
    seeds = sort!(collect(keys(l.data)))
    return (seed = seeds, y = [l.data[s] for s in seeds])
end
table_rows(::AverageLearner, table) = zip(table.seed, table.y)

# AverageLearner1D: one row per sample, sorted by (x, seed)
function to_table(l::AverageLearner1D)
    rows = [(seed, x, y) for (x, d) in l.samples for (seed, y) in d]
    sort!(rows; by = r -> (r[2], r[1]))
    return (seed = first.(rows), x = getindex.(rows, 2), y = last.(rows))
end
table_rows(::AverageLearner1D, table) = zip(zip(table.seed, table.x), table.y)

function to_table(l::IntegratorLearner)
    xs = sort!(collect(keys(l.data)))
    return (x = xs, y = [l.data[x] for x in xs])
end
table_rows(::IntegratorLearner, table) = zip(table.x, table.y)

# `tell!` only accepts points of intervals that exist, so replay by telling the queued points that are in the table
# and refining otherwise (upstream restores a pickled state instead).
function load_table!(l::IntegratorLearner, table)
    data = Dict(zip(Float64.(table.x), table.y))
    budget = 20 * length(data) + 100  # refinements allowed without a known point queued
    # ponytail: replays the learner's own refinement path; data that path never asks for are dropped (warned about). Upgrade: restore the full interval state as upstream.
    while length(l.data) < length(data)
        known = [x for x in l.stack if haskey(data, x)]
        foreach(x -> tell!(l, x, data[x]), known)
        isempty(known) || continue
        budget -= 1
        (budget > 0 && fill_stack!(l)) || break
    end
    length(l.data) < length(data) && @warn "only $(length(l.data)) of $(length(data)) points fit the learner's refinement path and were loaded"
    return l
end

# SequenceLearner
function to_table(l::SequenceLearner)
    idx = findall(l.told)
    return (index = idx, x = l.sequence[idx], y = l.values[idx])
end
table_rows(l::SequenceLearner, table) = zip(((i, l.sequence[i]) for i in table.index), table.y)
