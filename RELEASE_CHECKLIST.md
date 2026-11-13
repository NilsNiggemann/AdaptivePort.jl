# Release checklist

Steps for tagging a release. Nothing here is automated; registration is done by hand.

1. `main` is green on CI (tests on Julia 1.10 and 1, docs build).
2. Locally, with several threads:
   - `JULIA_NUM_THREADS=2 julia --project=. -e 'using Pkg; Pkg.test()'`
   - `julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'` and
     `julia --project=docs docs/make.jl` from inside the checkout.
   - Every script in `examples/` runs (`julia -t 4 --project=examples examples/<name>.jl`; the MPI example under
     `mpiexec -n 3`).
3. A clean environment loads the core without weak dependencies:
   `julia --project=$(mktemp -d) -e 'using Pkg; Pkg.develop(path="."); using AdaptivePort'`, and its Manifest
   holds none of Distributed, Dagger, MPI.
4. `Project.toml`: version bumped, `[compat]` covers every dependency and weak dependency, nothing unused.
5. `CHANGELOG.md`: move "unreleased" to the version and date; `docs/src/compatibility.md` matches the code.
6. `benchmark/README.md` rerun if a hot path changed.
7. Tag and register: comment `@JuliaRegistrator register` on the release commit on GitHub (needs the
   JuliaRegistrator app), then tag the commit `v<version>` once the registry PR merges (or add the TagBot workflow
   first, which the repo does not have yet).
