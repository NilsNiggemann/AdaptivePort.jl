# AverageLearner

[`AverageLearner`](@ref) estimates the mean of a noisy function. The function takes an integer
`seed` and returns one real sample; the seed must be its only source of randomness. The learner asks
for the seeds $0, 1, 2, \dots$ and stops when the standard error of the mean is within the tolerance.

```julia
using AdaptivePort, Random

noisy(seed) = 1.5 + randn(Random.Xoshiro(seed))
learner = AverageLearner(noisy; atol=0.05)
BlockingRunner(learner)            # stops by itself, no tol needed
result(learner)                    # (mean, std, n)
```

`examples/average_learner.jl` is the runnable script.

!!! danger "The seed must be the only source of randomness"
    `f(seed)` must return the same sample for the same seed and independent samples for different seeds. Seed
    a local generator from it, as above (`Random.Xoshiro(seed)`), never the global RNG: with the global RNG,
    parallel runs and resumed runs repeat or correlate samples, and the error estimate is wrong.

!!! tip "No `tol` needed"
    The learner has its own goal: `atol` (absolute) and/or `rtol` (relative) on the standard error of the mean.
    A runner without `tol` stops when it is reached.

## Advanced: algorithm

With $n$ samples, sample standard deviation $s$ and mean $\bar y$, the standard error is
$s / \sqrt{n}$. The loss is that error divided by `atol`, or by `rtol` times $\lvert \bar y \rvert$,
whichever is larger; it is `Inf` below `min_npoints` samples. So the loss is at most 1 exactly when
the mean is within tolerance, and `done(learner)` (without `tol`) tests `loss <= 1`. `ask` hands out
the smallest seeds that are neither told nor pending, so told seeds are never repeated and abandoned
seeds can be asked again after `remove_unfinished!`. `done(learner; tol)` compares the loss to `tol`
instead.

## Limits

Float64 only. The mean and variance come from running sums of $y$ and $y^2$ (as upstream), which
loses precision when the mean is much larger than the standard deviation. The standard error assumes
independent samples. Differences to upstream: [Deviations](../compatibility.md#Deviations-from-Python-Adaptive).

## API

```@docs
AverageLearner
```
