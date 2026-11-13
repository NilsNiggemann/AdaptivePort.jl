"""
    AverageLearner(f; atol = nothing, rtol = nothing, min_npoints = 2)

Learn the mean of a noisy function. `f(seed)` returns one real sample, and the integer `seed` is its only source of
randomness. The points the learner asks for are the seeds `0, 1, 2, ...`. Port of upstream's `AverageLearner`.

The loss is the standard error of the mean divided by `atol`, or divided by `rtol * abs(mean)`, whichever is larger. So
the loss is at most 1 when the error is within tolerance, and `done(learner)` means exactly that. It is `Inf` below
`min_npoints` samples.

**Keyword Arguments**
- `atol`: desired absolute tolerance of the mean.
- `rtol`: desired relative tolerance of the mean. At least one of `atol` and `rtol` is required.
- `min_npoints`: samples to take before the loss is finite; at least 2, because the standard deviation needs two.

**See Also**
- [`BlockingRunner`](@ref), [`result`](@ref)
"""
mutable struct AverageLearner{F} <: AbstractLearner
    const f::F
    const atol::Float64  # Inf if unused
    const rtol::Float64  # Inf if unused
    const min_npoints::Int
    const data::Dict{Int,Float64}
    const pending::Set{Int}
    sum_f::Float64  # ponytail: Float64 only; make it a type parameter T<:AbstractFloat if other precisions are needed
    sum_f_sq::Float64  # naive running sums as upstream; Welford's update if cancellation matters (mean much larger than std)
end

function AverageLearner(f; atol = nothing, rtol = nothing, min_npoints = 2)
    atol === nothing && rtol === nothing && throw(ArgumentError("at least one of `atol` and `rtol` must be set"))
    return AverageLearner(f, Float64(something(atol, Inf)), Float64(something(rtol, Inf)), max(min_npoints, 2), Dict{Int,Float64}(), Set{Int}(), 0.0, 0.0)
end

npoints(l::AverageLearner) = length(l.data)

function Base.show(io::IO, l::AverageLearner)
    print(io, "AverageLearner(", npoints(l), " points, ", length(l.pending), " pending)")
end

# Standard deviation estimate as upstream: Inf below `min_npoints`, corrected for the sample size.
function sample_std(l::AverageLearner)
    n = npoints(l)
    n < l.min_npoints && return Inf
    numerator = l.sum_f_sq - l.sum_f^2 / n
    numerator < 0 && return 0.0  # rounding error ~ -1e-15
    return sqrt(numerator / (n - 1))
end

# Loss if the standard deviation and mean stay as they are but there were `n` samples.
function loss_at(l::AverageLearner, n)
    n < l.min_npoints && return Inf
    standard_error = sample_std(l) / sqrt(n)
    mean = l.sum_f / npoints(l)
    aloss = standard_error / l.atol
    rloss = standard_error / l.rtol
    mean != 0 && (rloss /= abs(mean))
    return max(aloss, rloss)
end

"""
    loss(learner::AverageLearner; real = true)

Standard error of the mean in units of the tolerance (see [`AverageLearner`](@ref)). With `real = false` the pending
samples count towards the sample size.
"""
loss(l::AverageLearner; real = true) = loss_at(l, real ? npoints(l) : npoints(l) + length(l.pending))

"""
    done(learner::AverageLearner; tol = nothing)

Whether `loss(learner) <= tol`; without `tol`, whether the mean is within its tolerance (`loss(learner) <= 1`).
"""
done(l::AverageLearner; tol = nothing) = loss(l) <= something(tol, 1)

has_own_goal(::AverageLearner) = true

function ask_with_improvements(l::AverageLearner, n::Integer; tell_pending = true)
    points = Int[]
    seed = 0  # ponytail: O(npoints) scan from 0 per ask; keep a "first free seed" counter if runs reach ~1e5 samples
    while length(points) < n
        haskey(l.data, seed) || seed in l.pending || push!(points, seed)
        seed += 1
    end
    current = loss(l)
    improvement = isfinite(current) ? current - loss_at(l, npoints(l) + n) : Inf
    tell_pending && foreach(p -> tell_pending!(l, p), points)
    return points, fill(improvement / n, n)  # n == 0 gives an empty vector
end

function tell!(l::AverageLearner, seed::Integer, value::Real)
    haskey(l.data, seed) && return l  # already told, as upstream
    l.data[seed] = value
    delete!(l.pending, seed)
    l.sum_f += value
    l.sum_f_sq += value^2
    return l
end

tell_pending!(l::AverageLearner, seed::Integer) = (haskey(l.data, seed) || push!(l.pending, seed); l)

pending_points(l::AverageLearner) = sort!(collect(l.pending))

function remove_unfinished!(l::AverageLearner)
    empty!(l.pending)
    return l
end

"""
    result(learner::AverageLearner)

Named tuple `(mean, std, n)`: the sample mean, the corrected sample standard deviation (`Inf` below `min_npoints`
samples) and the number of samples. `mean` is `NaN` without samples.
"""
result(l::AverageLearner) = (mean = l.sum_f / npoints(l), std = sample_std(l), n = npoints(l))
