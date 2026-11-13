# Executors: where a runner evaluates the learner's function.
#
# Contract: an executor is any callable `executor(f, x)` that starts `f(x...)` somewhere and returns a handle `h`;
# `fetch(h)` blocks until the value is there and throws if `f` threw. The types below only add defaults
# (`default_ntasks`) and lifecycle; a plain function such as `(f, x) -> Threads.@spawn f(x...)` is a valid executor.

"""
    ThreadedExecutor()

Evaluate on the threads of this Julia process: `executor(f, x)` is `Threads.@spawn f(x...)`. The default executor of
the runners. Start Julia with several threads (`julia -t auto`) and pass `ntasks > 1` to evaluate concurrently.

`default_ntasks(ThreadedExecutor()) == 1`, so by default `f` runs one point at a time and need not be thread-safe.

**See Also**
- [`AsyncRunner`](@ref), [`DistributedExecutor`](@ref), [`DaggerExecutor`](@ref), [`MPIExecutor`](@ref)
"""
struct ThreadedExecutor end

(::ThreadedExecutor)(f, x) = Threads.@spawn f(x...)

"""
    DistributedExecutor(pool = CachingPool(workers()))

Evaluate on Distributed.jl worker processes: `executor(f, x)` is `remotecall(f, pool, x...)`. Needs `using
Distributed` (package extension). `default_ntasks` is the number of workers in `pool`.

`f` and everything it uses must be defined on the workers (`@everywhere`); only points and values travel. A
`CachingPool` sends a closure to each worker once instead of on every call.

**See Also**
- [`AsyncRunner`](@ref), [`ThreadedExecutor`](@ref)
"""
struct DistributedExecutor{P}
    pool::P
end

"""
    DaggerExecutor(; options...)

Evaluate through Dagger.jl's scheduler, which spreads work over the threads and Distributed workers it knows:
`executor(f, x)` is `Dagger.spawn(f, options, x...)`. Needs `using Dagger` (package extension).

Keywords are passed to `Dagger.Options` (e.g. `scope`) and apply to every evaluation. `default_ntasks` is
`Dagger.num_processors(scope)`, the threads of all processes Dagger knows. Workers added after Dagger's scheduler
started must be registered with `Dagger.addprocs!(Dagger.Sch.eager_context(), workers())`.

**See Also**
- [`AsyncRunner`](@ref), [`ThreadedExecutor`](@ref)
"""
struct DaggerExecutor{O}
    options::O
end

"""
    MPIExecutor(body, f; comm = MPI.COMM_WORLD)

Evaluate `f` on the ranks of an MPI job with native MPI.jl (package extension; needs `using MPI` and `MPI.Init()`).
All ranks run the same script: rank 0 runs `body(executor)` (typically a runner), the other ranks serve `f` until rank
0 leaves `body`. Only points and values are sent; every rank holds `f` itself, and the executor rejects any other
function. `default_ntasks` is the number of ranks minus one, so at least two ranks are needed.

Returns the value of `body` on rank 0 and `nothing` on the other ranks. Leaving `body` (normally or by an exception)
stops every worker after its current evaluation; results of abandoned evaluations are received and dropped. A crashed
rank is not detected and makes the run hang.

```julia
MPIExecutor(learner.f) do executor      # ranks > 0 serve f here and skip the block
    BlockingRunner(learner; tol = 1e-3, executor)
end
```

**See Also**
- [`AsyncRunner`](@ref), [`DistributedExecutor`](@ref)
"""
struct MPIExecutor{S}
    state::S
end

const EXTENSION_PACKAGES = ((DistributedExecutor, :Distributed), (DaggerExecutor, :Dagger), (MPIExecutor, :MPI))

# Readable error when an executor is used without its package loaded.
function executor_hint(io, exc, argtypes, kwargs)
    for (E, pkg) in EXTENSION_PACKAGES
        if exc.f === E || exc.f isa E
            ext = Base.get_extension(@__MODULE__, Symbol(:AdaptivePort, pkg, :Ext))
            ext === nothing && print(io, "\n$(nameof(E)) is defined in a package extension: run `using $pkg` first.")
        end
    end
end

function __init__()
    Base.Experimental.register_error_hint(executor_hint, MethodError)
end

"""
    captured_exception(e, backtrace)

The exception a runner reports in an [`EvaluationError`](@ref) when `fetch` on an evaluation handle throws `e`: the
original exception with the backtrace from inside the function where the handle provides it. Unexported; extensions
add methods for their remote exception types.
"""
captured_exception(e, bt) = CapturedException(e, bt)
captured_exception(e::TaskFailedException, bt) = CapturedException(e.task.exception, e.task.backtrace)
