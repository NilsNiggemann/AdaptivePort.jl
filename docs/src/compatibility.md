# Compatibility with Python Adaptive

One row per public item of upstream's `adaptive/__init__.py`, `adaptive.learner`, `adaptive.runner` and
`adaptive.utils`, and what it became here. Status words:

- **implemented**: same behaviour under the same (or Julia-cased) name.
- **equivalent**: same capability in the Julia idiom (`!`, keywords, multiple dispatch, Tables.jl, ...).
- **partial**: part of the behaviour only; the note says what is missing.
- **missing**: not ported; the note gives the workaround, if any.
- **deviation**: ported with an intentional difference (see [Deviations](#Deviations-from-Python-Adaptive)).

The behaviour is checked against upstream's `tests/test_learners.py` in `test/test_properties.jl` (see
[Property tests](#Property-tests)).

## Learners

| Upstream | Status | Note |
|:---|:---|:---|
| `BaseLearner` | equivalent | `AbstractLearner`; the contract is in its docstring. |
| `Learner1D` | implemented | Bounds are a tuple, `loss_per_interval` gets scaled coordinates, `vdim` for vector output. |
| `Learner2D` | partial | Delaunay triangulation via DelaunayTriangulation.jl, no point stack, no interpolator; see below. |
| `LearnerND` | partial | Own Bowyer-Watson triangulation; practical up to $N = 4$; no convex-hull bounds or interpolator. |
| `AverageLearner` | implemented | `result` is `(mean, std, n)`; integer seeds, smallest free seed first. |
| `AverageLearner1D` | implemented | Points are `(seed, x)`; `npoints` counts distinct `x`, `nsamples` all samples. |
| `IntegratorLearner` | implemented | `result` is `(integral, error, npoints)`; Float64 only. |
| `SequenceLearner` | deviation | Points are `(index, element)` with a 1-based index; `result` works before completion. |
| `BalancingLearner` | deviation | Same four strategies; 1-based child index; `from_product` is unexported. |
| `DataSaver` / `make_datasaver` | equivalent | `DataSaver(learner; arg_picker)`; `extra_data(saver)`; no `make_datasaver` (the constructor is the factory). |
| `SKOptLearner` | missing | Not in the upstream version this port follows (removed upstream); no scikit-optimize equivalent. |
| `learner.new()` | equivalent | Construct a new learner with the same arguments; saved data reload with [`load_data!`](@ref). |
| `learner.copy_from(other)` | missing | Use `load_table!(learner, to_table(other))` (unexported) or `deepcopy`. |

### Loss functions

| Upstream | Status | Note |
|:---|:---|:---|
| `learner1D.default_loss` | implemented | `default_loss(xs, ys)` on scaled coordinates (arc length). |
| `learner1D.uniform_loss`, `abs_min_log_loss` | missing | Pass any `(xs, ys) -> loss` function as `loss_per_interval`. |
| `learner1D.triangle_loss`, `curvature_loss_function`, `resolution_loss_function` | missing | Neighbour-based losses (`nth_neighbors`) are not supported by `Learner1D`. |
| `learner2D.default_loss` | deviation | `default_loss(points, values, triangles)` gets scaled points and raw values, not an interpolator. |
| `learner2D.uniform_loss`, `resolution_loss_function`, `minimize_triangle_surface_loss`, `thresholded_loss_function`, `triangle_loss` | missing | Pass any `(points, values, triangles) -> losses` as `loss_per_triangle`. |
| `learnerND.default_loss`, `uniform_loss`, `std_loss` | implemented | `default_loss_nd`, `uniform_loss_nd`, `std_loss_nd`. |
| `learnerND.triangle_loss`, `curvature_loss_function` | missing | Need neighbour simplices. |
| `LearnerND` `anisotropic`, `triangulation_backend`, convex-hull bounds | missing | Rectangular bounds only; `bounds` must be a box. |

## Learner methods and attributes

| Upstream | Status | Note |
|:---|:---|:---|
| `ask(n, tell_pending=True)` | deviation | `ask(learner, n; tell_pending = true)` returns the points only; `AdaptivePort.ask_with_improvements` gives upstream's pair. |
| `tell(x, y)` | equivalent | `tell!(learner, x, y)`. |
| `tell_many(xs, ys)` | equivalent | Loop over `tell!`; no optimized bulk path. |
| `tell_pending(x)` | equivalent | `tell_pending!(learner, x)`. |
| `remove_unfinished()` | equivalent | `remove_unfinished!(learner)`; for `IntegratorLearner` it re-queues the points. |
| `loss(real=True)` | equivalent | `loss(learner; real = true)`. |
| `npoints` | equivalent | `npoints(learner)`. |
| `done()` / `goal` | deviation | `done(learner; tol)`; no `tol` for learners with their own goal. |
| `result` / `learner.data` | equivalent | `result(learner)` returns copies as a named tuple. |
| `pending_points` | equivalent | `pending_points(learner)` returns a new `Vector`. |
| `to_dataframe` | equivalent | `to_table(learner)`, a Tables.jl column table; no pandas columns for function arguments. |
| `load_dataframe` | equivalent | `load_table!` (unexported) or `load_data!` from a file. |
| `save(fname)` / `load(fname)` | deviation | `save_data(path, learner)` / `load_data!(learner, path)`; CSV or `Serialization`, not pickle. |
| `plot()`, `interpolator`, `interpolated_on_grid` | missing | Plotting is docs-only recipes (CairoMakie); see [Plotting results](examples/plotting.md). `triangles` and `to_table` give the data. |
| `learner.function` | equivalent | The field `learner.f`. |
| `learner.bounds`, `learner.data` attributes | equivalent | Not guaranteed API; use `result`. |
| `AverageLearner1D.tell_many_at_point` | missing | Loop over `tell!`. |

## Runners

| Upstream | Status | Note |
|:---|:---|:---|
| `Runner` | deviation | One `Runner` type; the loop runs in an `@async` task and evaluations in spawned tasks. |
| `BlockingRunner` | deviation | A function building a `Runner` and waiting; it needs a goal unless the learner has its own. |
| `AsyncRunner` | deviation | A function building a `Runner`; `wait`, `status`, `cancel!`. |
| `goal` | partial | Keywords `tol`, `max_npoints`, `duration`, `end_time`; no custom callable. `stop_after`, `auto_goal` become keywords. |
| `executor` | deviation | Callable `executor(f, x)` returning a `fetch`-able handle: `ThreadedExecutor`, `DistributedExecutor`, `DaggerExecutor`, `MPIExecutor`. |
| `ntasks` | deviation | Keyword `ntasks`; default `1` for `ThreadedExecutor` (upstream: all cores). |
| `retries`, `raise_if_retries_exceeded` | missing | The first failure stops the runner with an `EvaluationError`. |
| `log`, `replay_log` | missing | No event log. |
| `shutdown_executor` | missing | Executors are not owned by the runner. |
| `overhead()`, `elapsed_time()` | missing | Measure with `time()` around `wait(runner)`. |
| `live_info`, `live_info_terminal`, `live_plot` | missing | `status(runner)` gives the state; no notebook integration. |
| `start_periodic_saving` | equivalent | Runner keywords `checkpoint` and `checkpoint_interval`; one more save when it stops. |
| `simple(learner, goal)` | equivalent | `BlockingRunner(learner; ...)` with `ntasks = 1`. |
| `runner.task`, `runner.failed`, `runner.tracebacks`, `runner.to_retry` | partial | `wait(runner)` rethrows the `EvaluationError`; no tracebacks list. |
| `notebook_extension`, `active_plotting_tasks` | missing | Not applicable. |

## Utilities

| Upstream | Status | Note |
|:---|:---|:---|
| `utils.save`, `utils.load` | deviation | [`save_data`](@ref), [`load_data!`](@ref). |
| `utils.named_product` | equivalent | `AdaptivePort.from_product` builds a balancer over a product of parameters. |
| `utils.restore` | missing | Not needed: `ask(...; tell_pending = false)` leaves the learner unchanged. |
| `utils.daskify`, `SequentialExecutor`, `cache_latest`, `copy_docstring_from`, `assign_defaults`, `partial_function_from_dataframe` | missing | Python-specific or tied to pandas. |

## Property tests

`test/test_properties.jl` ports the generic tests of upstream's `tests/test_learners.py` and runs each over the
learners upstream parametrizes it with.

| Upstream test | Learners | Status |
|:---|:---|:---|
| `test_adding_existing_data_is_idempotent` | 1D, 2D, ND (2D and 3D), Sequence, Average1D | ported |
| `test_adding_non_chosen_data` | 1D, ND, Average, Average1D, Sequence, and 2D (upstream xfail) | ported; 2D passes because it has no point stack |
| `test_point_adding_order_is_irrelevant` | 1D, Average, Average1D | ported; for `Learner1D` the bounds are told first, because the lazy y-scale (upstream's default `_recompute_losses_factor = 2`, which upstream's test sets to 1) depends on the order in which the y-range grows |
| the same, 2D and ND | | left out (upstream xfail: the loss of `Learner2D` depends on the triangle order; here too) |
| `test_expected_loss_improvement_is_less_than_total_loss` | 1D, ND, Average, Average1D (and 2D, xfail) | ported as: no improvement exceeds the loss; for `AverageLearner` the sum is below the loss |
| `test_learner_performance_is_invariant_under_scaling` | 1D, ND, Average1D | ported with an asymmetric function (exactly symmetric functions create ties that rounding breaks differently) |
| `test_balancing_learner` | 1D, 2D, ND, Average, Average1D, Sequence | ported with pending and late points |
| `test_saving`, `test_saving_of_balancing_learner`, `test_saving_with_datasaver` | all | ported for CSV and `Serialization`; `Learner1D` and `AverageLearner1D` agree on the loss up to the lazy y-scale (a factor 2), not exactly |
| `ask(...; tell_pending = false)` leaves the learner unchanged | all | added (upstream documents it, no generic test) |
| `test_uniform_sampling*`, `test_learner_accepts_lists`, `test_convergence_for_arbitrary_ordering`, `test_learner_subdomain`, `test_to_dataframe` | | left out (upstream xfail, Python-specific or covered by the learner tests) |

## Deviations from Python Adaptive

These are intentional for now and may be revisited.

All learners:

- `ask` returns only the points; `AdaptivePort.ask_with_improvements` (unexported) gives upstream's pair.
- `vdim` is fixed at construction. Vector values are `NTuple`s, and `tell!` throws `ArgumentError`
  for a value of the wrong length.
- `tell!` with a point outside the bounds throws everywhere; upstream stores and ignores it.
- `tell!` of data that is already there is idempotent or an error, as upstream.
- `save_data` / `load_data!` write CSV or `Serialization` files of `to_table(learner)`, not pickles of the whole
  learner; the function is supplied by the constructor, pending points are not saved, and `load_data!` throws
  `ArgumentError` for a missing file (upstream's `load` ignores it silently). After a reload the data are identical, but
  where the state depends on the history (the lazy y-scale of `Learner1D` and `AverageLearner1D`) later decisions and
  the loss can differ by the factor-2 y-scale rule.
- `loss(learner; real = false)` counts pending points as if evaluated.

`Learner1D`:

- Non-finite components are excluded from the $y$-range per component; upstream includes `inf`.
  The internal `AdaptivePort.update_value!` never shrinks the range, as upstream.
- `ask` bisects; upstream splits one interval into `k` equal parts when it receives several points.
- `done(learner; tol)` exists; upstream expresses goals through runners (`loss_goal`).
- Not implemented: neighbour-based losses (`nth_neighbors`, curvature) and optimized bulk `tell_many`.

`LearnerND` (see [its page](learners/learnernd.md)):

- No convex-hull bounds, interpolator, plotting, `anisotropic`, `triangulation_backend` or
  neighbour-based losses (`triangle_loss`, `curvature_loss_function`).
- Non-finite values throw `ArgumentError`. `loss` is `Inf` until all corners are told.
- `loss(real = false)`: a simplex with pending points counts as its largest sub-simplex (upstream returns the real loss).
- `remove_unfinished!` rebuilds the simplex queue; upstream can lose subdivided simplices.
- The random start point uses `Xoshiro(1)`, so it differs from upstream.
- `ask(...; tell_pending = false)` snapshots the learner and restores it.
- The triangulation supports dimension 1 and builds the initial triangulation by insertion instead
  of scipy (the same result for points in general position). A failed insertion throws and leaves
  the triangulation unchanged. Asked points are clamped to the box.

`AverageLearner` (see [its page](learners/average_learner.md)):

- `ask` hands out the smallest seeds that are neither told nor pending; upstream starts at
  told plus pending and may skip a free low seed.
- `result` is `(mean, std, n)`. Float64 only.

`AverageLearner1D` (see [its page](learners/average_learner1d.md)):

- An undersampled $x$ is picked as the smallest such $x$; upstream picks an arbitrary set element.
- Pending bounds are excluded from re-asking; upstream re-asks a pending bound.
- The resample exclusion (`error <= min_error` or `nsamples >= max_samples`) is evaluated on demand,
  not dropped permanently.
- `max_samples` is not enforced for `ask(l, n > 1)` (upstream neither).
- The loss improvement of a resample uses the real neighbour losses.
- `npoints` counts distinct $x$ and `nsamples` all samples, so a runner's `max_npoints` counts
  distinct $x$.
- No `tell_many_at_point`, plotting or dataframes.

`IntegratorLearner` (see [its page](learners/integrator.md)):

- `result` is `(integral, error, npoints)`. Float64 only.
- `ask(...; tell_pending = false)` re-queues the asked points (no snapshot); `tell!` drops told
  points from the queue. `tell_pending!` is a no-op, as upstream. `remove_unfinished!` re-queues the
  asked but untold points (upstream: no-op, so abandoned points are never asked again).
- `DivergentIntegralError` propagates unchanged from `tell!` and `ask`; upstream converts it in `ask`.
- `ask` returns fewer than `n` points when the intervals run out and throws only when it has none.
- Reloading a saved learner replays the points in the saved order, which may drop points that came from a parallel run
  (they refer to intervals that the replay no longer has); the reloaded integral is then at least as accurate as its
  reported error.
- Kept upstream limitation: if values are told out of order, or asked in batches (as the parallel
  runner does), `done` can fire with an error estimate well above `tol` (upstream issue #96).
  Example: `exp` on $(0, 1)$, `tol = 1e-10`, batch size 8 gives an error estimate of $1.4 \cdot
  10^{-6}$ at 40 points, identical to upstream. The result is still within its reported error.

`SequenceLearner` (see [its page](learners/sequence.md)):

- The index is 1-based.
- `result` is `(indices, values)` of the told elements and works before completion; upstream raises
  until done and returns the values only.
- `value_type` keyword; no dataframes. CSV export fails for non-numeric elements; use a `.jls` path.

`BalancingLearner`:

- The child index in a point `(index, point)` is 1-based; `from_product` is unexported
  (`AdaptivePort.from_product`).
- A balancer is saved as one file (its `to_table` holds the child index); upstream writes one file per child through a
  `fname` function.
- `tell_pending!` invalidates both cached losses, upstream only the real one.
- `ask(...; tell_pending = false)` leaves the children untouched (for an `IntegratorLearner` child, earlier pending
  points are re-queued).

`DataSaver`:

- `extra_data` is an unordered `Dict`, upstream an `OrderedDict`; a repeated point overwrites its stored result.
- `to_table` carries an `extra_data` column, so saving needs a `.jls` path.

Runners:

- One `Runner` type; `BlockingRunner` and `AsyncRunner` are functions that construct and start it.
- Executors are callables `executor(f, x)` returning a `fetch`-able handle, not `concurrent.futures`
  executors. The default `ThreadedExecutor` runs with `ntasks = 1` (upstream: all cores), so `f`
  need not be thread-safe unless `ntasks` is raised.
- Goals: `tol`, `max_npoints`, `duration`, `end_time`; no custom `goal` callable. `npoints` plus the
  evaluations in flight never exceeds `max_npoints`, so it is never overshot.
- Errors: no `retries`; the first failure stops the runner with an `EvaluationError` after telling
  the results that already arrived. Upstream wraps it in a `RuntimeError`.
- Learners with their own goal stop without `tol`, and `BlockingRunner` accepts them without any goal.
- No logging, `overhead`, `elapsed_time` or live plotting. Periodic saving is `checkpoint`.
- The runner loop is the only caller of `ask` and `tell!`; it asks with `tell_pending = false`.
- Plotting is documented as recipes only (no Makie dependency, no `plot` methods).

`Learner2D` additionally:

- has no persistent point stack (`stack_size`): every `ask` recomputes the triangle losses, which
  equals upstream with `stack_size = 1`. Upstream's default (10) hands out points chosen from an
  older triangulation.
- may pick a different diagonal for cocircular points (e.g. the four corners), so point sequences
  differ from upstream after the corners, while the losses reached are comparable.
- ships only `default_loss`; `uniform_loss`, `resolution_loss_function`,
  `minimize_triangle_surface_loss` and `thresholded_loss_function` are not ported (pass any
  `(points, values, triangles) -> losses` function as `loss_per_triangle`).
- does not provide an interpolator (`ip()`) or `interpolated_on_grid`.
- ignores non-finite values in the triangulation; upstream propagates them into the loss.
