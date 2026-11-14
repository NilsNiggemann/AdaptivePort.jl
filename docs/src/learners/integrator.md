# IntegratorLearner

[`IntegratorLearner`](@ref) computes $\int_a^b f(x) \thinspace dx$ by adaptive quadrature. It is a port of
upstream's learner, which implements Gonnet's doubly adaptive Clenshaw-Curtis scheme
(algorithm 4 of "Increasing the Reliability of Adaptive Quadrature Using Explicit Partitioning").
Points are plain numbers $x$.

```julia
using AdaptivePort

f(x) = x^-0.5                       # integrable singularity, the integral is 2
learner = IntegratorLearner(f; bounds=(0, 1), tol=1e-6)
BlockingRunner(learner)             # stops on done(learner), no tol for the runner
result(learner)                     # (integral, error, npoints)
```

`examples/integrator.jl` also shows a manual `ask`/`tell!` loop.

## Advanced: algorithm

Each interval holds a Clenshaw-Curtis rule of 5, 9, 17 or 33 nodes, so refining an interval reuses
all points evaluated before. The error estimate of an interval is the $L_2$ distance of two
successive interpolants times the interval width. The interval with the largest error is refined to
the next rule, or bisected when it already has 33 nodes or when refining did not reduce its error.
The integral is the sum over the intervals whose points are all told; the error is the sum of their
estimates. Values may be told in any order and some may be skipped. Non-finite values are removed
from the interpolant (downdating) instead of failing; intervals that are too narrow, or whose error
reached machine accuracy, are dropped and their error is remembered. At most `max_ivals` (1000)
intervals are kept.

`done(learner)` holds once the error is below `tol` times $\lvert\text{integral}\rvert$ (or all
remaining error stems from intervals that cannot be refined). `ask` returns the queued points and
refines on demand; it returns fewer than $n$ points when the intervals run out, and throws only if
it has none.

## Limits

!!! warning "Parallel runs can stop early"
    When values are told out of order, or asked in batches as a parallel runner does, `done` can fire with an
    error estimate well above `tol` (upstream issue #96). For $e^x$ on $(0, 1)$ with `tol = 1e-10` and batch size
    8 the error estimate is $1.4 \cdot 10^{-6}$ at 40 points, as upstream. The result is still within its
    *reported* error: check `result(learner).error`.

- Float64 only.
- `DivergentIntegralError` is thrown by `tell!` or `ask` when the integral appears to diverge.
- `tell_pending!` and `remove_unfinished!` do nothing, as upstream.

Further differences: [Deviations](../compatibility.md#Deviations-from-Python-Adaptive).

## API

```@docs
IntegratorLearner
DivergentIntegralError
```
