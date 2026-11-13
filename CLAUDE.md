# AdaptivePort.jl — instructions for assistants

Julia port of [Python Adaptive](https://github.com/python-adaptive/adaptive). Staged plans live in
`.ignore/plans/planNN.md` (git-ignored); follow the current one.

## Layout
- `src/AdaptivePort.jl` — module, exports, includes, the `AbstractLearner` contract. One file per concept:
  `learner1D.jl`, `learner2D.jl` + `triangulation.jl`, `learnerND.jl` + `triangulationND.jl` (own
  Bowyer–Watson port), `average_learner.jl`, `average_learner1D.jl`, `integrator_learner.jl` +
  `integrator_coeffs.jl`, `sequence_learner.jl`, `runner.jl`.
- `test/runtests.jl` includes `test/test_<concept>.jl`.
- `examples/` — runnable scripts: `julia --project=. examples/learner1d.jl`.
- `docs/` — Documenter (`docs/src/index.md` plus one page per Stage-4 learner); build with `julia --project=docs docs/make.jl`
  (needs `Pkg.develop(path=".")` in the docs env, as CI does). Build from inside the git checkout,
  otherwise Documenter fails to find the remote. `makedocs` errors on any docstring (including the
  module's) missing from an `@docs` block.

## Conventions
- Follow the `julia-style` skill. No dependencies unless they clearly pay for themselves.
- Upstream behaviour is the reference, not its architecture. Read the upstream source when
  porting (clone `python-adaptive/adaptive` into the scratchpad; it is not installed locally).
- Record every intentional deviation from upstream in the "Deviations from Python Adaptive"
  section of `docs/src/index.md`.
- Deliberate shortcuts carry a `# ponytail:` comment naming the ceiling and upgrade path.
- Test: `julia --project=. -e 'using Pkg; Pkg.test()'`. Interactive checks: julia-repl MCP
  (activate the repo path; Revise is loaded, call `revise` after editing `src/`; changing a struct
  layout needs `reset`).
- Losses follow upstream's contract: `loss_per_interval(xs, ys)` gets *scaled* coordinates
  (`x / (b - a)`, `y / yscale`). Do not reintroduce unscaled heuristics like `dx * (1 + abs(dy))`:
  the `dy` term vanishes as `dx -> 0` and sampling degenerates to a uniform grid.
- Check adaptivity with a steep function (`tanh(50x)`), not only with smooth ones.

## Learner interface
`AbstractLearner` (docstring in `src/AdaptivePort.jl` lists the contract): `ask(learner, n;
tell_pending=true) -> Vector`, `tell!(learner, x, y)`, `tell_pending!`, `loss(learner)`, `npoints`,
`result`; `done(learner; tol)` is generic. The learner never calls the function; the caller
evaluates; runners call `learner.f(point...)`. 2D/ND points are `NTuple{N,T}` throughout.
- Points per learner: `AverageLearner` integer seed (from 0), `AverageLearner1D` `(seed, x)`,
  `IntegratorLearner` `x`, `SequenceLearner` `(index, element)` (1-based; `f` field is a wrapper
  ignoring the index). `npoints` of `AverageLearner1D` counts distinct `x` (`nsamples` all samples).
- Stage-5 interface: `pending_points`, `remove_unfinished!`, `loss(l; real = false)` and unexported
  `ask_with_improvements` (generic `ask` returns its first element). Implemented by `LearnerND`,
  `AverageLearner`, `AverageLearner1D`, `IntegratorLearner`, `SequenceLearner`; `Learner1D`/`Learner2D`
  get them in Stage 5.
- Own goal: `AdaptivePort.has_own_goal(l)` (default `false`; true for `AverageLearner`,
  `IntegratorLearner`, `SequenceLearner`) with `done(l; tol = nothing)`; the runner stops on `done(l)`
  when no `tol` is given.
- Vector output: keyword `vdim = 1` in `Learner1D`/`Learner2D`/`LearnerND`; `vdim = M > 1` stores
  values as `NTuple{M,T}` (value type is the second struct parameter), `tell!` takes any iterable of
  length `M`. Scalar path unchanged.
- `Learner2D`: triangulation only via `src/triangulation.jl` (DelaunayTriangulation.jl, full
  rebuild per `ask`/`loss`). `loss_per_triangle(points, values, triangles)` gets *scaled* points
  (domain → `[-1/2,1/2]^2`) and raw values. `estimate_gradients` is a scipy port checked to 1e-14;
  compare against upstream with `uv run --no-project --with adaptive python`.

## Stage status
- Stage 1 (`Learner1D`, plan01): done. Open deviations are listed in `docs/src/index.md`.
- Stage 2 (`Learner2D`, plan02): done.
- Stage 3 (runners, plan03): done. `src/runner.jl`: one `Runner` (loop in an `@async` task sticky to the
  caller's thread, evaluations in `Threads.@spawn`ed tasks), built by `BlockingRunner`/`AsyncRunner`.
  Only the runner loop calls `ask`/`tell!`; it asks with `tell_pending=false`.
- Stage 4 (plan04): done. Vector output, `LearnerND`, `AverageLearner`, `AverageLearner1D`,
  `IntegratorLearner`, `SequenceLearner`; deps `LinearAlgebra`, `StatsFuns` (Student-t quantile).
  `LearnerND` is practical up to N = 4 (linear-scan point location).
