## Dagger.jl

`DaggerExecutor` sends every evaluation to Dagger's scheduler (`Dagger.spawn(f, options, x...)`), which places it on
a thread or worker process it knows. Use it when you already use Dagger, or want threads and Distributed workers
behind one executor. Load the package first (`using Dagger`, package extension).

```julia
using Distributed, AdaptivePort
addprocs(2)
@everywhere using Dagger
using Dagger
Dagger.addprocs!(Dagger.Sch.eager_context(), workers())
learner = Learner1D(x -> tanh(50x); bounds = (-1, 1))
BlockingRunner(learner; tol = 1e-2, executor = DaggerExecutor())
```

- Keywords go to `Dagger.Options`, e.g. `DaggerExecutor(; scope = Dagger.ProcessScope(2))` restricts evaluation to
  worker 2. `default_ntasks` is `Dagger.num_processors(scope)`.
- Dagger does not notice workers added after its scheduler context was created; call
  `Dagger.addprocs!(Dagger.Sch.eager_context(), workers())` (or add workers before first use). Otherwise tasks fail
  with a `SchedulingException`.
- `f` must be defined on every worker that may run it (`@everywhere`).
- A throwing `f` yields an `EvaluationError` holding the original exception and its backtrace from inside `f`.
  There are no retries.
