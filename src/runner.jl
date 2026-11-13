"""
    EvaluationError{P} <: Exception

Thrown by a runner when the learner's function throws at `point`.

Fields: `point::P` (the input that failed) and `exception::CapturedException` (the original exception with the
backtrace from inside the function).

**See Also**
- [`AsyncRunner`](@ref), [`BlockingRunner`](@ref)
"""
struct EvaluationError{P} <: Exception
    point::P
    exception::CapturedException
end

function Base.showerror(io::IO, e::EvaluationError)
    print(io, "EvaluationError: the learner's function failed at ", repr(e.point), ":\n")
    showerror(io, e.exception)
end

"""
    Runner{L<:AbstractLearner}

Drives a learner: repeatedly asks it for a batch of points, evaluates the learner's function `learner.f` on them (`f(x)` in 1D, `f(x, y)` in 2D) and tells
the results back. Created by [`AsyncRunner`](@ref) or [`BlockingRunner`](@ref); it starts on construction.

Fields: `learner` (inspect progress with `npoints(runner.learner)` and `loss(runner.learner)`), `cancelled` (set by
[`cancel!`](@ref)) and `task` (the `Task` running the loop).

**Type Parameters**
- `L<:AbstractLearner`: the learner type; the runner uses only the [`AbstractLearner`](@ref) interface.

**See Also**
- [`status`](@ref), [`cancel!`](@ref), `wait(runner)`
"""
mutable struct Runner{L<:AbstractLearner}
    const learner::L
    @atomic cancelled::Bool
    task::Task
    Runner(learner::L) where {L<:AbstractLearner} = new{L}(learner, false)
end

"""
    AsyncRunner(learner; tol = nothing, max_npoints = nothing, batch_size = parallel ? Threads.nthreads() : 1, parallel = false)

Start learning `learner.f` in the background and return the [`Runner`](@ref) immediately.

Each iteration asks for `batch_size` points with `tell_pending = false`, evaluates them, and tells every successful result
back. The loop stops when `loss(learner) <= tol`, when `npoints(learner) >= max_npoints`, when `ask` returns no points
(the learner is resolved to its numerical limit), after [`cancel!`](@ref), or when an evaluation throws.

The loop runs in a task pinned to the caller's thread (`@async`), and the function evaluations run in tasks spawned on
other threads. So the learner is only mutated on the caller's thread, at the caller's yield points (`sleep`, `wait`, I/O),
and reading `runner.learner` from the calling task is safe without locks. Reading it from another thread is not.

**Keyword Arguments**
- `tol`: stop once `loss(learner) <= tol` (upstream `loss_goal`). `nothing` disables it.
- `max_npoints`: stop once the learner holds this many points (upstream `npoints_goal`); no batch exceeds it, so it is a
  hard limit on evaluations. `nothing` disables it. With neither goal the runner runs until `cancel!` or until `ask`
  returns nothing.
- `batch_size`: points requested per iteration. `1` is the most adaptive; larger batches only pay off with `parallel`.
- `parallel`: evaluate the points of a batch concurrently, one task each (`Threads.@spawn`). Start Julia with several
  threads (`julia -t auto`), and `learner.f` must be safe to call concurrently.

**Returns**
- A running [`Runner`](@ref). `wait(runner)` blocks until it stops and rethrows a failure.

**Errors**
- If `learner.f` throws, the other points of that batch are still evaluated and told, then the runner fails with an
  [`EvaluationError`](@ref) holding the failed point. Nothing is left pending in the learner. No retries.

**See Also**
- [`BlockingRunner`](@ref), [`status`](@ref), [`cancel!`](@ref)
"""
function AsyncRunner(learner::AbstractLearner; tol = nothing, max_npoints = nothing, parallel = false, batch_size = parallel ? Threads.nthreads() : 1)
    batch_size >= 1 || throw(ArgumentError("batch_size must be positive, got $batch_size"))
    r = Runner(learner)
    # @async, not Threads.@spawn: sticky to the caller's thread, so the learner never races with the caller.
    r.task = @async run!(r, tol, max_npoints, batch_size, parallel)
    return r
end

"""
    BlockingRunner(learner; tol = nothing, max_npoints = nothing, batch_size, parallel = false)

Same as [`AsyncRunner`](@ref), but return only once the runner has stopped. Rethrows an [`EvaluationError`](@ref).
At least one of `tol` and `max_npoints` is required, as upstream refuses to block forever.

**Returns**
- The finished [`Runner`](@ref).
"""
function BlockingRunner(learner::AbstractLearner; tol = nothing, max_npoints = nothing, kwargs...)
    if tol === nothing && max_npoints === nothing
        throw(ArgumentError("BlockingRunner needs a goal: pass tol and/or max_npoints"))
    end
    return wait(AsyncRunner(learner; tol, max_npoints, kwargs...))
end

goal_reached(l, tol, max_npoints) = (tol !== nothing && loss(l) <= tol) || (max_npoints !== nothing && npoints(l) >= max_npoints)

function try_evaluate(f, x)
    try
        return f(x...)  # splat: a 2D point (x, y) calls f(x, y); a number splats to itself
    catch e
        return EvaluationError(x, CapturedException(e, catch_backtrace()))
    end
end

function evaluate(f, xs, parallel)
    parallel || return map(x -> try_evaluate(f, x), xs)
    return fetch.([Threads.@spawn(try_evaluate(f, x)) for x in xs])
end

# The runner loop; returns the final status, `:finished` or `:cancelled`.
function run!(r::Runner, tol, max_npoints, batch_size, parallel)
    l = r.learner
    while true
        @atomic(r.cancelled) && return :cancelled
        goal_reached(l, tol, max_npoints) && return :finished
        n = max_npoints === nothing ? batch_size : min(batch_size, max_npoints - npoints(l))
        # tell_pending = false: a failed or cancelled batch leaves no stale pending points behind.
        xs = ask(l, n; tell_pending = false)
        isempty(xs) && return :finished
        # ponytail: batch-synchronous, fast points wait for the slowest one; refill per finished point (upstream) if evaluation times vary widely
        ys = fetch(Threads.@spawn evaluate(l.f, xs, parallel))
        for (x, y) in zip(xs, ys)
            y isa EvaluationError || tell!(l, x, y)
        end
        i = findfirst(y -> y isa EvaluationError, ys)
        i === nothing || throw(ys[i])
    end
end

"""
    wait(runner::Runner)

Block until `runner` stops. Rethrows the exception that stopped it (an [`EvaluationError`](@ref) for a failed evaluation),
unwrapped from Julia's `TaskFailedException`.

**Returns**
- `runner`.
"""
function Base.wait(r::Runner)
    try
        wait(r.task)
    catch e
        e isa TaskFailedException ? throw(e.task.exception) : rethrow()
    end
    return r
end

"""
    status(runner)

The runner's state: `:running`, `:finished` (goal reached or nothing left to ask), `:cancelled` or `:failed`.

There is no "created" state, because runners start on construction, as upstream.
"""
function status(r::Runner)
    istaskdone(r.task) || return :running
    return istaskfailed(r.task) ? :failed : fetch(r.task)
end

"""
    cancel!(runner)

Ask `runner` to stop. Takes effect between batches: the batch being evaluated finishes and is told to the learner first,
so the learner is never left with unevaluated pending points. `wait(runner)` returns once it has stopped.

**Returns**
- `runner`.
"""
function cancel!(r::Runner)
    @atomic r.cancelled = true
    return r
end
