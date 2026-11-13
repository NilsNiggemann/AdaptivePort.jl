# AverageLearner1D

[`AverageLearner1D`](@ref) learns a noisy scalar function on an interval. It decides between two
things: sample a new $x$, or draw more samples at an existing $x$. Points are tuples `(seed, x)` and
the function is called as `f(seed, x)`; it must return one sample whose randomness comes from `seed`
only.

```julia
using AdaptivePort, Random

noisy(seed, x) = tanh(10x) + (0.05 + 0.2 * (x > 0)) * randn(Random.Xoshiro(hash((seed, x))))
learner = AverageLearner1D(noisy; bounds=(-1, 1), min_samples=10)
BlockingRunner(learner; max_npoints=40)   # counts distinct x, not samples

r = result(learner)    # (x, y, error, nsamples), sorted by x
nsamples(learner)      # all samples; npoints(learner) counts distinct x
```

`examples/average_learner1d.jl` prints how the samples spread: the noisy right half gets far more
samples than the quiet left half.

## Algorithm

An inner [`Learner1D`](@ref) holds the sample means and chooses where a new $x$ goes, with the usual
interval loss. Each $x$ also has an `error`: the half-width of the confidence interval of its mean,
using Student's $t$ quantile at level $1 - \alpha$ (`alpha`), so the true value lies in
$\bar y \pm \text{error}$ with probability $1 - 2\alpha$. It is `Inf` with one sample.

`ask(l, n)` proposes all $n$ points at one $x$:

1. If some $x$ is undersampled (fewer than `min_samples` samples, or fewer than `neighbor_sampling`
   times the mean sample count of its neighbours) it gets $n$ more seeds.
2. With at most one $x$ told, a new $x$ is proposed.
3. Otherwise the $x$ with the largest rescaled error (its `error` over the smaller adjacent segment
   length of the means) is resampled if that exceeds `delta`; else a new $x$ is chosen by the
   `Learner1D` rule on the means.

An $x$ with `error <= min_error` or `max_samples` samples is not resampled by rule 3. `ask` never
modifies the learner itself; pending points are registered through `tell_pending!`. The loss is the
loss of the means; `loss(learner; real = false)` lets pending $x$ split their interval.

## Limits

Resampling cost is a linear scan over the $x$ per `ask`. `max_samples` is not enforced when
`ask(l, n)` is called with $n > 1$ (as upstream). Differences to upstream:
[Deviations](index.md#Deviations-from-Python-Adaptive).

## API

```@docs
AverageLearner1D
nsamples
```
