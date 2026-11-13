# Benchmarks

Run: `julia -t 4 --project=benchmark benchmark/benchmarks.jl` (optional arguments: `quick`, `trace`, and any of
`learners runner parallel balancing save` to run only those sections). The environment `benchmark/` has
BenchmarkTools and the package (`Pkg.develop`); it is not a dependency of the package.

## Setup of the reported run

- CPU: AMD EPYC 9454 48-Core Processor, shared login machine with a load average of about 67 (timings are noisy,
  expect +-20 %), Julia 1.13.0, 4 threads (`-t 4`), 2 Distributed workers for the Distributed row.
- Commit: 2a44483 (main at the time) plus the two optimizations below.
- Method: wall time (`@elapsed`) of one run after a warm-up; `f` is `tanh(50x) + sin(3x)` (1D) or
  `tanh(50xy) + sin(3x)` (2D), the learners are driven by a hand-written loop `ask(l, 1)`, `tell!`. "us/point" is the
  total divided by N, so it is an average over a growing learner.

## Results

| Benchmark | Before | After |
|---|---|---|
| `Learner1D` ask/tell, N = 100 | 0.2 ms | 0.4 ms (noise) |
| `Learner1D` ask/tell, N = 1000 | 14.4 ms | 9.3 ms |
| `Learner1D` ask/tell, N = 10000 | 2.59 s (259 us/point) | 0.91 s (91 us/point) |
| `Learner2D` ask/tell, N = 100 | 77 ms | 33 ms |
| `Learner2D` ask/tell, N = 1000 | 8.13 s (8.1 ms/point) | 4.57 s (4.6 ms/point) |
| `Learner2D` ask/tell, N = 3000 | not measured (too slow) | 46 s (15 ms/point) |
| `Learner1D` `loss()` at N = 10000 | | 3 us |
| `Learner2D` `loss()` at N = 1000 | | 10 ms |
| 1D hand loop, trivial `f`, N = 5000 | | 48 us/point |
| 1D `BlockingRunner`, trivial `f`, N = 5000 | | 59 us/point (runner overhead 10 us/point) |
| 10 ms `f`, N = 100, sequential | | 1.01 s |
| same, threaded (`ntasks = 4`) | | 0.25 s, 4.0x |
| same, Distributed (2 workers) | | 0.55 s, 1.85x |
| `BalancingLearner` | not present in this checkout, not benchmarked | |
| `save_data` N = 1e6 `.csv` / `.jls` | | 0.99 s / 0.09 s |
| `load_data!` N = 1e6 `.csv` / `.jls` | | 0.64 s / 0.12 s |

## What was optimized

Both changes keep the points identical: a fixed sequence of `ask` (with and without `tell_pending`), `tell_pending!`
and `tell!` for 335 points returns the same points and the same `loss` (real and not real) before and after, for
`Learner1D` and `Learner2D` (checked by hashing the printed points).

- `Learner2D`: `combined_data` triangulated the evaluated points to interpolate the pending and chosen points even
  when there are none. Every `ask` from a runner (`tell_pending = false`, nothing in flight at the ask) therefore
  triangulated twice. It now skips the first triangulation when there is nothing to interpolate: 8.1 s to 4.6 s at
  N = 1000. The remaining cost is one Delaunay rebuild per `ask` (7 ms at N = 1000) and one per `loss`.
- `Learner1D`: `combined_intervals` located the interval of every point with `searchsortedlast`, O(N log N) per `ask`.
  The positions are monotone, so a moving index does it in O(N): 475 us to 60 us per `ask` at N = 1e4, 2.6 s to 0.9 s
  for a 1e4 point run.

Measured and left alone: the per-point runner overhead (10 us, task and channel), `loss`, `tell!` (6 us at N = 1e4),
`save_data`. Type stability: `@inferred` succeeds for `ask`, `ask_with_improvements` and `tell!` of both learners and
for `loss` of `Learner1D`; `loss(::Learner2D)` is not inferred (it goes through DelaunayTriangulation.jl and a
user-supplied `loss_per_triangle`), which is invisible next to the 7 ms triangulation, so it was not touched.

Known remaining ceilings (see the `# ponytail:` comments): the O(N) interval rebuild in `Learner1D`, and the full
triangulation per `ask` in `Learner2D` (incremental `DT.add_point!` would remove it).
