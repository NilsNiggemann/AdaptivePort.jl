"""
    AverageLearner1D{T, F, I}

Adaptive sampler of a noisy scalar function on an interval: it decides between sampling a new `x` and drawing more samples at an existing `x`. Port of upstream `AverageLearner1D`.

A point is a tuple `(seed, x)`, the runner calls `f(seed, x)`, and `f` must return one noisy sample that depends on `seed` only through its randomness. Samples with the same `seed` at the same `x` are the same sample, so a repeated one is ignored.

**Sampling rule.** [`ask`](@ref) proposes all `n` points at one `x`, because `n` is normally much smaller than `min_samples`. If some point is undersampled (the smallest such `x`; upstream picks an arbitrary one) it gets `n` more seeds, continuing after the samples told so far. With at most one told `x`, a new `x` is proposed. Otherwise the `x` with the largest rescaled error (its `error` over the smaller adjacent segment length) is resampled if that exceeds `delta`, and else a new `x` is chosen by the `Learner1D` rule on the means. `ask` never changes the learner itself; with `tell_pending = true` the points are registered with [`tell_pending!`](@ref).

An inner [`Learner1D`](@ref) holds the sample means and decides where to put a new `x`; its stored mean is moved with the unexported `update_value!` whenever a sample arrives. The y-scale of the inner learner is widened by every individual sample, not only by the means, as upstream.

Fields: `f`, `inner` (the `Learner1D` of means), `samples` (`x => (seed => y)`), `undersampled` (points with fewer than `min_samples` samples, or fewer than `neighbor_sampling` times the mean count of their neighbors), `error` (half-width of the confidence interval of the mean at each `x`, `Inf` with one sample), `pending` (issued `(seed, x)` not yet told) and the settings `delta`, `alpha`, `neighbor_sampling`, `min_samples`, `max_samples`, `min_error`.

**Type Parameters**
- `T<:AbstractFloat`: type of `x` and of the sample values.
- `I<:Learner1D`: the inner learner.

Construct with [`AverageLearner1D(f; bounds)`](@ref AverageLearner1D).

**See Also**
- [`Learner1D`](@ref), [`nsamples`](@ref)
"""
mutable struct AverageLearner1D{T<:AbstractFloat, F, I<:Learner1D} <: AbstractLearner
    f::F
    inner::I
    samples::Dict{T, Dict{Int, T}}
    undersampled::Set{T}
    error::Dict{T, T}
    pending::Set{Tuple{Int, T}}
    delta::T
    alpha::T
    neighbor_sampling::T
    min_samples::Int
    max_samples::Int
    min_error::T
end

"""
    AverageLearner1D(f; bounds = (-1, 1), loss_per_interval = default_loss, delta = 0.2, alpha = 0.005,
                     neighbor_sampling = 0.3, min_samples = 50, max_samples = typemax(Int), min_error = 0)

Create an empty learner for the noisy function `f(seed, x)` on the interval `bounds`.

**Keyword Arguments**
- `bounds`: `(a, b)`, as for [`Learner1D`](@ref).
- `loss_per_interval`: loss of one interval of the means, as for [`Learner1D`](@ref) (scaled coordinates).
- `delta`: resampling condition, `0 < delta <= 1`. A point is resampled if its uncertainty is larger than `delta` times the smaller of the two neighboring segment lengths (unscaled distance in the x-y plane of the means).
- `alpha`: `0 < alpha <= 1`. The true value at `x` lies within `mean ± error` with probability `1 - 2 alpha`. `0.005` is recommended.
- `neighbor_sampling`: `0 < neighbor_sampling <= 1`. A new point is first sampled at least this fraction of the average number of samples of its neighbors.
- `min_samples`: every new point is first sampled at least this many times.
- `max_samples`: no point gets resampled by the uncertainty rule once it has this many samples; `min_samples <= max_samples`.
- `min_error`: a point whose `error` is at most this is not resampled any more.

**Returns**
- An [`AverageLearner1D`](@ref); throws `ArgumentError` for invalid settings.

**See Also**
- [`ask`](@ref), [`tell!`](@ref), [`result`](@ref)
"""
function AverageLearner1D(f; bounds::Tuple{Real, Real} = (-1, 1), loss_per_interval = default_loss, delta::Real = 0.2, alpha::Real = 0.005, neighbor_sampling::Real = 0.3, min_samples::Integer = 50, max_samples::Integer = typemax(Int), min_error::Real = 0)
    0 < delta <= 1 || throw(ArgumentError("requires 0 < delta <= 1, got $delta"))
    0 < alpha <= 1 || throw(ArgumentError("requires 0 < alpha <= 1, got $alpha"))
    0 < neighbor_sampling <= 1 || throw(ArgumentError("requires 0 < neighbor_sampling <= 1, got $neighbor_sampling"))
    min_samples >= 0 || throw(ArgumentError("min_samples must not be negative, got $min_samples"))
    min_samples <= max_samples || throw(ArgumentError("max_samples must be at least min_samples"))
    inner = Learner1D(f; bounds, loss_per_interval)
    T = typeof(inner.dx_eps)
    return AverageLearner1D(f, inner, Dict{T, Dict{Int, T}}(), Set{T}(), Dict{T, T}(), Set{Tuple{Int, T}}(), T(delta), T(alpha), T(neighbor_sampling), Int(min_samples), Int(max_samples), T(min_error))
end

"""
    npoints(learner::AverageLearner1D)

Number of distinct `x` with at least one sample; [`nsamples`](@ref) counts all samples. A runner's `max_npoints` counts this.
"""
npoints(l::AverageLearner1D) = length(l.inner.xs)

"""
    nsamples(learner)

Total number of samples told to `learner`, summed over all `x`.

**See Also**
- [`npoints`](@ref), [`AverageLearner1D`](@ref)
"""
nsamples(l::AverageLearner1D) = sum(length, values(l.samples); init = 0)

"""
    result(learner::AverageLearner1D)

Evaluated data as a named tuple of vectors sorted by `x`: `x`, `y` (the means), `error` (half-width of the confidence interval of the mean, `Inf` for one sample) and `nsamples` (samples at each `x`).
"""
function result(l::AverageLearner1D)
    xs = copy(l.inner.xs)
    return (x = xs, y = copy(l.inner.ys), error = [l.error[x] for x in xs], nsamples = [length(l.samples[x]) for x in xs])
end

pending_points(l::AverageLearner1D) = sort!(collect(l.pending); by = p -> (p[2], p[1]))

function remove_unfinished!(l::AverageLearner1D)
    empty!(l.pending)
    empty!(l.inner.pending)
    return l
end

# Largest loss over the intervals of xs and pending, a pending point splitting its interval proportionally (upstream losses_combined).
function combined_loss(l::AverageLearner1D{T}) where {T}
    inner = l.inner
    xs = inner.xs
    all(b -> insorted(b, xs) || insorted(b, inner.pending), inner.bounds) || return T(Inf)
    combined = sort!(vcat(xs, inner.pending))
    worst = T(-Inf)
    for k in 1:length(combined)-1
        a, b = combined[k], combined[k+1]
        j = searchsortedlast(xs, a)
        if 1 <= j < length(xs) && b <= xs[j+1]
            dx = xs[j+1] - xs[j]
            worst = max(worst, priority(inner, inner.losses[j], dx) * (b - a) / dx)
        else
            worst = T(Inf)
        end
    end
    return worst == -Inf ? T(Inf) : worst
end

"""
    loss(learner::AverageLearner1D; real = true)

Loss of the means as for [`Learner1D`](@ref); `Inf` until both bounds are told. With `real = false` pending points count as split points of their interval, as upstream, and an interval next to a pending point without a told neighbor has loss `Inf`.
"""
loss(l::AverageLearner1D; real = true) = real ? loss(l.inner) : combined_loss(l)

function tell_pending!(l::AverageLearner1D, p::Tuple{Integer, Real})
    seed, x = Int(p[1]), oftype(l.inner.dx_eps, p[2])
    push!(l.pending, (seed, x))
    insorted(x, l.inner.xs) || tell_pending!(l.inner, x)
    return l
end

"""
    tell!(learner::AverageLearner1D, (seed, x), y)

Record the sample `y = f(seed, x)`. A new `x` is inserted into the inner learner with `y` as its mean; a new `seed` at a known `x` updates the mean and `error` there; a known `(seed, x)` is ignored. Throws `ArgumentError` for `x` outside the bounds.

**See Also**
- [`tell_pending!`](@ref), [`result`](@ref)
"""
function tell!(l::AverageLearner1D{T}, p::Tuple{Integer, Real}, y::Real) where {T}
    seed, x, yt = Int(p[1]), T(p[2]), T(y)
    inner = l.inner
    delete!(l.pending, (seed, x))
    if !haskey(l.samples, x)
        tell!(inner, x, yt)
        l.samples[x] = Dict(seed => yt)
        push!(l.undersampled, x)
        l.error[x] = T(Inf)
    elseif !haskey(l.samples[x], seed)
        resample!(l, x, seed, yt)
    end
    return l
end

function resample!(l::AverageLearner1D{T}, x, seed, y) where {T}
    inner = l.inner
    samples = l.samples[x]
    i = searchsortedfirst(inner.xs, x)
    n = length(samples)
    new_mean = inner.ys[i] * n / (n + 1) + y / (n + 1)
    samples[seed] = y
    n += 1
    if x in l.undersampled && n >= l.min_samples
        neighbors = [length(l.samples[inner.xs[j]]) for j in (i - 1, i + 1) if 1 <= j <= length(inner.xs)]
        mean_neighbors = isempty(neighbors) ? 0 : sum(neighbors) / length(neighbors)
        n > l.neighbor_sampling * mean_neighbors && delete!(l.undersampled, x)
    end
    # Student-t factor makes the true value lie within mean ± error with probability 1 - 2 alpha
    variance_in_mean = sum(v -> (v - new_mean)^2, values(samples)) / (n - 1)
    l.error[x] = StatsFuns.tdistinvcdf(n - 1, 1 - l.alpha) * sqrt(variance_in_mean / n)
    # upstream widens the y-scale with every individual sample, the inner learner only sees the means
    inner.yrange = widen_range(inner.yrange, y)
    update_value!(inner, x, new_mean)
    return nothing
end

# Error of the mean at x relative to the smaller adjacent segment (unscaled x-y distance), upstream rescaled_error.
function rescaled_error(l::AverageLearner1D, i)
    xs, ys = l.inner.xs, l.inner.ys
    d_left = i > 1 ? hypot(xs[i] - xs[i-1], ys[i] - ys[i-1]) : nothing
    d_right = i < length(xs) ? hypot(xs[i+1] - xs[i], ys[i+1] - ys[i]) : nothing
    return l.error[xs[i]] / min(something(d_left, d_right), something(d_right, d_left))
end

# Point with the largest rescaled error among those still allowed more samples; `nothing` if none.
function most_uncertain(l::AverageLearner1D)
    best, best_value = nothing, -Inf
    for (i, x) in enumerate(l.inner.xs)
        (l.error[x] <= l.min_error || length(l.samples[x]) >= l.max_samples) && continue
        value = rescaled_error(l, i)
        # ponytail: O(npoints) scan per ask where upstream keeps a sorted dict, use a heap if npoints reaches ~1e4
        if value > best_value
            best, best_value = x, value
        end
    end
    return best === nothing ? (nothing, best_value) : (best, best_value)
end

function more_samples(l::AverageLearner1D{T}, x, n) where {T}
    inner = l.inner
    n_existing = length(l.samples[x])
    i = searchsortedfirst(inner.xs, x)
    loss_left = i > 1 ? inner.losses[i-1] : T(Inf)
    loss_right = i < length(inner.xs) ? inner.losses[i] : T(Inf)
    loss = (loss_left + loss_right) / 2
    improvement = isinf(loss) ? T(Inf) : loss - loss * sqrt(n_existing) / sqrt(n_existing + n)
    return [(seed + n_existing, x) for seed in 0:n-1], fill(improvement / n, n)
end

function new_point(l::AverageLearner1D{T}, n) where {T}
    xs = ask(l.inner, 1; tell_pending = false)
    isempty(xs) && return Tuple{Int, T}[], T[]
    improvement = combined_loss(l) / 2
    return [(seed, only(xs)) for seed in 0:n-1], fill(improvement / n, n)
end

# Sampling rule, see the AverageLearner1D docstring. ask itself never changes the learner; tell_pending registers the points.
function ask_with_improvements(l::AverageLearner1D{T}, n::Integer; tell_pending = true) where {T}
    n < 0 && throw(ArgumentError("n must be non-negative, got $n"))
    n == 0 && return Tuple{Int, T}[], T[]
    # ponytail: ask does not mutate the learner (inner.ask with tell_pending = false), so no snapshot is needed
    if !isempty(l.undersampled)
        points, improvements = more_samples(l, minimum(l.undersampled), n)
    elseif npoints(l) <= 1
        points, improvements = new_point(l, n)
    else
        x, value = most_uncertain(l)
        if x !== nothing && value > l.delta
            points, improvements = more_samples(l, x, n)
        else
            points, improvements = new_point(l, n)
        end
    end
    tell_pending && foreach(p -> tell_pending!(l, p), points)
    return points, improvements
end
