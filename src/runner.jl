"""
    EvaluationError{P} <: Exception

Thrown by a runner when the learner's function throws at `point`.

Fields: `point::P` (the input that failed) and `exception::CapturedException` (the original exception, with the
backtrace from inside the function where the executor provides it).

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

# Messages to the runner loop. Results come from the evaluation tasks, wake-ups from `cancel!` and the timers.
struct Outcome{P, Y}
    point::P
    value::Y  # the function value, or an `EvaluationError`
end

struct Wakeup
    reason::Symbol  # :cancel, :deadline or :checkpoint
end

"""
    Runner{L<:AbstractLearner}

Drives a learner: keeps up to `ntasks` evaluations of `learner.f` in flight (as `f(x...)`: `f(x)` in 1D, `f(x, y)` in
2D), and asks the learner for a new point as soon as one finishes. Created by [`AsyncRunner`](@ref) or
[`BlockingRunner`](@ref); it starts on construction.

Fields: `learner` (inspect progress with `npoints(runner.learner)` and `loss(runner.learner)`), `events` (the channel
the loop waits on), `cancelled` (set by [`cancel!`](@ref)) and `task` (the `Task` running the loop).

**Type Parameters**
- `L<:AbstractLearner`: the learner type; the runner uses only the [`AbstractLearner`](@ref) interface.

**See Also**
- [`status`](@ref), [`cancel!`](@ref), `wait(runner)`
"""
mutable struct Runner{L<:AbstractLearner}
    const learner::L
    const events::Channel{Any}
    @atomic cancelled::Bool
    task::Task
    Runner(learner::L) where {L<:AbstractLearner} = new{L}(learner, Channel{Any}(Inf), false)
end

# Stopping criteria of one run; `deadline` is in `time()` seconds.
struct Goals{T, N}
    tol::T
    max_npoints::N
    deadline::Float64
end

seconds(d::Real) = Float64(d)
seconds(d::Dates.Period) = Dates.toms(d) / 1000
seconds(d::Dates.CompoundPeriod) = Dates.toms(d) / 1000

function deadline_of(duration, end_time)
    t = Inf
    duration === nothing || (t = min(t, time() + seconds(duration)))
    end_time === nothing || (t = min(t, time() + seconds(end_time - Dates.now())))
    return t
end

"""
    AsyncRunner(learner; tol = nothing, max_npoints = nothing, duration = nothing, end_time = nothing,
                executor = ThreadedExecutor(), ntasks = default_ntasks(executor),
                checkpoint = nothing, checkpoint_interval = 30)

Start learning `learner.f` in the background and return the [`Runner`](@ref) immediately.

The runner keeps `ntasks` evaluations in flight. Whenever results arrive it tells them to the learner, checks the
goals, and asks for as many new points as there are free slots (with `tell_pending = true`), so a slow evaluation
never holds back the others (upstream behaviour).

It stops when `loss(learner) <= tol` (without `tol`: when `done(learner)`, for learners with their own stopping
criterion), when `npoints(learner) >= max_npoints`, at the deadline, when `ask` returns no points and nothing is in
flight (the learner is resolved to its numerical limit), after [`cancel!`](@ref), or when an evaluation throws. On
every stop, results that have already arrived are told, evaluations still running are abandoned (they cannot be
interrupted; their results are dropped), and `remove_unfinished!(learner)` clears the pending points.

The loop runs in a task pinned to the caller's thread (`@async`) and is the only code that touches the learner, so the
learner is mutated on the caller's thread at the caller's yield points (`sleep`, `wait`, I/O). Reading
`runner.learner` from the calling task is safe without locks; reading it from another thread is not.

**Keyword Arguments**
- `tol`: stop once `loss(learner) <= tol` (upstream `loss_goal`). `nothing` disables it.
- `max_npoints`: stop once the learner holds this many points (upstream `npoints_goal`). `npoints` plus the
  evaluations in flight never exceeds it, so it is a hard limit on evaluations.
- `duration`: stop this many seconds (a `Real` or a `Dates.Period`) after construction (upstream `duration_goal`).
- `end_time`: stop at this `Dates.DateTime` (local time, as `Dates.now()`; upstream `end_time_goal`).
- `executor`: where `f` runs; any callable `executor(f, x)` returning a `fetch`-able handle (see
  [`ThreadedExecutor`](@ref), [`DistributedExecutor`](@ref), [`DaggerExecutor`](@ref), [`MPIExecutor`](@ref)).
- `ntasks`: evaluations kept in flight. The default for [`ThreadedExecutor`](@ref) is `1`, so `f` runs sequentially
  and need not be thread-safe; use `ntasks = Threads.nthreads()` for a threaded run.
- `checkpoint`: a file path; the runner calls [`save_data`](@ref)`(checkpoint, learner)` every
  `checkpoint_interval` seconds and once more when it stops for any reason, including a failure.
- `checkpoint_interval`: seconds between checkpoints.

Without `tol`, `max_npoints`, `duration` and `end_time`, the runner runs until `cancel!` or until `ask` returns nothing.

**Returns**
- A running [`Runner`](@ref). `wait(runner)` blocks until it stops and rethrows a failure.

**Errors**
- If `learner.f` throws, results that already arrived are told and the runner fails with an
  [`EvaluationError`](@ref) holding the failed point. No retries.

**See Also**
- [`BlockingRunner`](@ref), [`status`](@ref), [`cancel!`](@ref)
"""
function AsyncRunner(learner::AbstractLearner; tol = nothing, max_npoints = nothing, duration = nothing, end_time = nothing, executor = ThreadedExecutor(), ntasks::Integer = default_ntasks(executor), checkpoint = nothing, checkpoint_interval::Real = 30)
    ntasks >= 1 || throw(ArgumentError("ntasks must be positive, got $ntasks"))
    checkpoint_interval > 0 || throw(ArgumentError("checkpoint_interval must be positive, got $checkpoint_interval"))
    goals = Goals(tol, max_npoints, deadline_of(duration, end_time))
    r = Runner(learner)
    # @async, not Threads.@spawn: sticky to the caller's thread, so the learner never races with the caller.
    r.task = @async run!(r, goals, executor, Int(ntasks), checkpoint, Float64(checkpoint_interval))
    return r
end

"""
    BlockingRunner(learner; kwargs...)

Same as [`AsyncRunner`](@ref) (same keywords), but return only once the runner has stopped. Rethrows an
[`EvaluationError`](@ref). At least one of `tol`, `max_npoints`, `duration` and `end_time` is required, as upstream
refuses to block forever, unless the learner has its own stopping criterion (`AverageLearner`, `IntegratorLearner`,
`SequenceLearner`).

**Returns**
- The finished [`Runner`](@ref).
"""
function BlockingRunner(learner::AbstractLearner; tol = nothing, max_npoints = nothing, duration = nothing, end_time = nothing, kwargs...)
    if all(isnothing, (tol, max_npoints, duration, end_time)) && !has_own_goal(learner)
        throw(ArgumentError("BlockingRunner needs a goal: pass tol, max_npoints, duration and/or end_time"))
    end
    return wait(AsyncRunner(learner; tol, max_npoints, duration, end_time, kwargs...))
end

function goal_reached(l, g::Goals)
    time() >= g.deadline && return true
    g.max_npoints !== nothing && npoints(l) >= g.max_npoints && return true
    g.tol === nothing && return has_own_goal(l) && done(l)
    return loss(l) <= g.tol
end

# Start one evaluation. The wrapper task makes any `fetch`-able handle work and reports the outcome to the loop.
function launch(r::Runner, executor, f, x)
    @async begin
        y = try
            fetch(executor(f, x))
        catch e
            EvaluationError(x, captured_exception(e, catch_backtrace()))
        end
        put!(r.events, Outcome(x, y))
    end
    return nothing
end

# The runner loop; returns the final status, `:finished` or `:cancelled`, or throws the first `EvaluationError`.
function run!(r::Runner, g::Goals, executor, ntasks, checkpoint, interval)
    l = r.learner
    timers = Timer[]
    isfinite(g.deadline) && push!(timers, Timer(_ -> put!(r.events, Wakeup(:deadline)), max(0.0, g.deadline - time())))
    checkpoint === nothing || push!(timers, Timer(_ -> put!(r.events, Wakeup(:checkpoint)), interval; interval))
    inflight = 0
    failure = nothing
    try
        while true
            @atomic(r.cancelled) && return :cancelled
            failure === nothing || throw(failure)
            goal_reached(l, g) && return :finished
            nfree = ntasks - inflight
            g.max_npoints === nothing || (nfree = min(nfree, g.max_npoints - npoints(l) - inflight))
            if nfree > 0
                for x in ask(l, nfree)
                    launch(r, executor, l.f, x)
                    inflight += 1
                end
            end
            inflight == 0 && return :finished
            # Block for one event, then drain whatever else is ready, so a burst of results is told at once.
            ev = take!(r.events)
            while true
                if ev isa Outcome
                    inflight -= 1
                    if ev.value isa EvaluationError
                        failure === nothing && (failure = ev.value)
                    else
                        tell!(l, ev.point, ev.value)
                    end
                elseif ev.reason === :checkpoint
                    save_data(checkpoint, l)
                end
                isready(r.events) || break
                ev = take!(r.events)
            end
        end
    finally
        foreach(close, timers)
        remove_unfinished!(l)
        checkpoint === nothing || save_data(checkpoint, l)
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

The runner's state: `:running`, `:finished` (a goal or the deadline was reached, or nothing was left to ask),
`:cancelled` or `:failed`.

There is no "created" state, because runners start on construction, as upstream.
"""
function status(r::Runner)
    istaskdone(r.task) || return :running
    return istaskfailed(r.task) ? :failed : fetch(r.task)
end

"""
    cancel!(runner)

Stop `runner` as soon as its loop wakes up, which it does immediately even while evaluations are running. Results that
have already arrived are told; evaluations still running are abandoned and their points removed from the learner's
pending points. `wait(runner)` returns once it has stopped.

**Returns**
- `runner`.
"""
function cancel!(r::Runner)
    @atomic r.cancelled = true
    put!(r.events, Wakeup(:cancel))
    return r
end
