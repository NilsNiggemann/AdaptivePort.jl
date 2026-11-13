# AdaptivePort.jl — instructions for assistants

Julia port of [Python Adaptive](https://github.com/python-adaptive/adaptive). Staged plans live in
`.ignore/plans/planNN.md` (git-ignored); follow the current one.

## Layout
- `src/AdaptivePort.jl` — module, exports, includes. One file per concept (`src/learner1D.jl`).
- `test/runtests.jl` includes `test/test_<concept>.jl`.
- `examples/` — runnable scripts: `julia --project=. examples/learner1d.jl`.
- `docs/` — Documenter (`docs/src/index.md`); build with `julia --project=docs docs/make.jl`
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

## Learner interface (as of stage 1, informal — no abstract type yet)
`ask(learner, n; tell_pending=true) -> Vector`, `tell!(learner, x, y)`, `tell_pending!`,
`loss(learner)`, `done(learner; tol)`, `npoints`, `result`. The learner never calls the function;
the caller evaluates. Introduce `AbstractLearner` only when the second learner arrives.

## Stage status
- Stage 1 (`Learner1D`, plan01): done. Open deviations are listed in `docs/src/index.md`.
