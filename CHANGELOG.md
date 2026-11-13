# Changelog

## 0.1.0 (unreleased)

First release. A Julia implementation of [Python Adaptive](https://github.com/python-adaptive/adaptive):
learners choose where to evaluate an expensive function through an `ask`/`tell!` interface, and runners evaluate
it on threads, Distributed workers, Dagger or MPI ranks. The [compatibility page](docs/src/compatibility.md) lists
which upstream items exist and every deviation.

### Learners
- `Learner1D`, `Learner2D`, `LearnerND` (practical up to N = 4), with upstream's loss functions and vector output
  (`vdim`).
- `AverageLearner`, `AverageLearner1D` for noisy functions; `IntegratorLearner` (Gonnet's doubly adaptive
  Clenshaw–Curtis quadrature); `SequenceLearner`.
- `BalancingLearner` over several learners of one kind, with the strategies `:loss_improvements`, `:loss`,
  `:npoints` and `:cycle`; `AdaptivePort.from_product` builds one per parameter combination.
- `DataSaver` keeps the full result of each evaluation (`extra_data`) while the child learns a part of it.
- Shared interface: `pending_points`, `remove_unfinished!`, `loss(learner; real = false)` and the unexported
  `ask_with_improvements`.

### Runners and executors
- `BlockingRunner` and `AsyncRunner` refill a slot as soon as one evaluation finishes, with at most `ntasks`
  evaluations in flight. Goals: `tol`, `max_npoints`, `duration`, `end_time`, or the learner's own goal.
  `status`, `cancel!`, errors reported as `EvaluationError`.
- Executors are callables `executor(f, x)` returning a fetchable handle: `ThreadedExecutor` (default) and, as
  package extensions, `DistributedExecutor` (`using Distributed`), `DaggerExecutor` (`using Dagger`) and
  `MPIExecutor` (`using MPI`). A missing extension gives an error hint naming the package.
- Periodic and final checkpoints with `checkpoint = path` and `checkpoint_interval`.

### Data
- `to_table` (named tuple of columns), `save_data` and `load_data!`: CSV for numeric tables, `Serialization`
  (`.jls`, or any other extension) for everything else; writes are atomic.

### Known limitations
- No retries and no custom goal callable; abandoned evaluations keep running after a stop.
- `.jls` files depend on the Julia version and the package's types; use CSV to exchange data.
- An `IntegratorLearner` reloaded from a table replays its refinement path and may drop points that a parallel
  run evaluated off that path.
- A crashed MPI rank is not detected and makes the run hang. Dagger on MPI is not covered.
- `Learner2D` rebuilds its Delaunay triangulation on every `ask` (about 7 ms at 1000 points).
