module AdaptivePortMPIExt

using AdaptivePort, MPI

const TASK_TAG = 1
const RESULT_TAG = 2
const LOST_TAG = 3  # a worker leaves `serve` early (exit, uncaught error): (id or 0, CapturedException)

# Handle of one evaluation: `fetch` blocks until the poller has stored the result and rethrows a worker failure.
mutable struct MPIHandle
    const done::Base.Event
    value::Any
    ok::Bool
    MPIHandle() = new(Base.Event(), nothing, true)
end

function finish!(h::MPIHandle, ok, value)
    h.ok = ok
    h.value = value
    notify(h.done)
    return h
end

function Base.fetch(h::MPIHandle)
    wait(h.done)
    return h.ok ? h.value : throw(h.value)
end

"""
Wraps the `CapturedException` of a failed remote evaluation, so that `captured_exception` can unwrap it.
"""
struct MPIRemoteError <: Exception
    captured::CapturedException
end

Base.showerror(io::IO, e::MPIRemoteError) = showerror(io, e.captured)

AdaptivePort.captured_exception(e::MPIRemoteError, bt) = e.captured

lost_error(rank, reason) = MPIRemoteError(CapturedException(ErrorException("MPI rank $rank was lost: $reason"), backtrace()))

# State of the manager (rank 0). Only touched from tasks on the thread that created it, under `lock` for safety.
mutable struct MPIState{F, C}
    const f::F
    const comm::C
    const timeout::Float64                        # seconds per evaluation before its rank counts as lost
    const lock::ReentrantLock
    const idle::Vector{Int}                       # ranks waiting for a point
    const busy::Dict{Int, Tuple{Int, Float64}}    # rank -> (id of the point it evaluates, start time)
    const lost::Set{Int}                          # ranks that left `serve` early or timed out; get no new points
    const timedout::Set{Int}                      # lost ranks that may still be alive; a late result revives them
    const queue::Vector{Tuple{Int, Any}}          # points waiting for an idle rank
    const handles::Dict{Int, MPIHandle}
    const requests::Vector{MPI.AbstractRequest}   # sends in flight (they keep their buffers alive)
    const stray::Vector{MPI.AbstractRequest}      # stop messages to timed-out ranks; never waited for
    nextid::Int
    closed::Bool
end

function MPIState(f, comm, timeout)
    return MPIState(f, comm, Float64(timeout), ReentrantLock(), collect(1:MPI.Comm_size(comm)-1),
                    Dict{Int, Tuple{Int, Float64}}(), Set{Int}(), Set{Int}(), Tuple{Int, Any}[],
                    Dict{Int, MPIHandle}(), MPI.AbstractRequest[], MPI.AbstractRequest[], 0, false)
end

nworkers(s::MPIState) = MPI.Comm_size(s.comm) - 1

function post!(s::MPIState, msg, rank, tag)
    push!(s.requests, MPI.isend(msg, rank, tag, s.comm))
    return nothing
end

function fail!(s::MPIState, id, err)
    h = pop!(s.handles, id, nothing)
    h === nothing || finish!(h, false, err)
    return nothing
end

# Hand queued points to idle ranks; fail them all once no rank is left.
function dispatch!(s::MPIState)
    while !s.closed && !isempty(s.queue) && !isempty(s.idle)
        rank = popfirst!(s.idle)
        id, x = popfirst!(s.queue)
        s.busy[rank] = (id, time())
        post!(s, (id, x), rank, TASK_TAG)
    end
    if length(s.lost) == nworkers(s)
        for (id, _) in s.queue
            fail!(s, id, MPIRemoteError(CapturedException(ErrorException("every MPI worker rank was lost"), backtrace())))
        end
        empty!(s.queue)
    end
    return nothing
end

# Mark `rank` lost and fail the evaluation it held.
function lose!(s::MPIState, rank, err)
    push!(s.lost, rank)
    filter!(!=(rank), s.idle)
    busy = pop!(s.busy, rank, nothing)
    busy === nothing || fail!(s, first(busy), err)
    return nothing
end

# Receive every result and loss report that has arrived, and time out slow ranks; returns whether anything changed.
function receive!(s::MPIState)
    got = false
    while true
        isready, status = MPI.Iprobe(MPI.ANY_SOURCE, RESULT_TAG, s.comm, MPI.Status)
        isready || break
        got = true
        rank = status.source
        id, ok, value = MPI.recv(rank, RESULT_TAG, s.comm, nothing)
        delete!(s.busy, rank)
        if rank in s.timedout  # it answered late: its point has failed already, but the rank can work again
            delete!(s.timedout, rank)
            delete!(s.lost, rank)
        end
        rank in s.lost || s.closed || push!(s.idle, rank)
        h = pop!(s.handles, id, nothing)
        h === nothing || finish!(h, ok, ok ? value : MPIRemoteError(value))
    end
    while true
        isready, status = MPI.Iprobe(MPI.ANY_SOURCE, LOST_TAG, s.comm, MPI.Status)
        isready || break
        got = true
        rank = status.source
        _, captured = MPI.recv(rank, LOST_TAG, s.comm, nothing)
        lose!(s, rank, MPIRemoteError(captured))
    end
    if isfinite(s.timeout)
        now = time()
        for (rank, (_, t0)) in collect(s.busy)
            if now - t0 > s.timeout
                got = true
                lose!(s, rank, lost_error(rank, "its evaluation exceeded the timeout of $(s.timeout) s"))
                push!(s.timedout, rank)
            end
        end
    end
    filter!(r -> !MPI.Test(r), s.requests)
    return got
end

function poll!(s::MPIState)
    lock(s.lock) do
        got = receive!(s)
        dispatch!(s)
        got
    end
end

function poller(s::MPIState)
    while !s.closed
        poll!(s) || sleep(0.001)
    end
end

function (e::AdaptivePort.MPIExecutor{<:MPIState})(f, x)
    s = e.state
    f === s.f || throw(ArgumentError("MPIExecutor evaluates only the function it was created with; got a different one"))
    h = MPIHandle()
    lock(s.lock) do
        s.closed && throw(ErrorException("MPIExecutor is closed"))
        s.nextid += 1
        s.handles[s.nextid] = h
        push!(s.queue, (s.nextid, x))
        dispatch!(s)
    end
    return h
end

AdaptivePort.default_ntasks(e::AdaptivePort.MPIExecutor{<:MPIState}) = nworkers(e.state)

# Stop all live workers, receive (and drop) the results of evaluations still running, fail every unfinished handle.
# A rank that left `serve` gets no stop message; a timed-out rank gets one that is not waited for (a send to a dead
# rank may never complete), so that it exits once its evaluation ends.
function close!(s::MPIState)
    lock(s.lock) do
        s.closed = true
        empty!(s.queue)
        for rank in 1:nworkers(s)
            if rank in s.timedout
                push!(s.stray, MPI.isend(nothing, rank, TASK_TAG, s.comm))
            elseif !(rank in s.lost)
                post!(s, nothing, rank, TASK_TAG)
            end
        end
    end
    while true
        lock(s.lock) do
            receive!(s)
        end
        isempty(s.busy) && break
        sleep(0.001)
    end
    foreach(MPI.Wait, s.requests)
    empty!(s.requests)
    for h in values(s.handles)
        finish!(h, false, ErrorException("MPIExecutor stopped before the evaluation finished"))
    end
    empty!(s.handles)
    isempty(s.timedout) || @warn "MPI rank(s) $(sort!(collect(s.timedout))) timed out and were abandoned; if one hangs, it keeps the MPI job alive: end the job with `MPI.Abort(comm, 1)`"
    return nothing
end

# Tell rank 0 that this worker leaves `serve` without its stop message, so its point fails instead of hanging.
function report_lost(comm, id, err, bt)
    MPI.Finalized() && return nothing
    MPI.send((id, CapturedException(err, bt)), 0, LOST_TAG, comm)
    return nothing
end

# Worker: serve `f` until the stop message. A failing evaluation is reported and the loop goes on. Leaving early (an
# error outside `f`, or `exit` called by `f`) is reported to rank 0 as a lost rank.
function serve(f, comm)
    current = Ref(0)    # id of the point being evaluated, 0 between points
    active = Ref(true)  # false once this `serve` has ended (normally or with its loss reported)
    atexit() do
        active[] || return nothing
        active[] = false
        err = ErrorException("MPI rank $(MPI.Comm_rank(comm)) was lost: the process exited during the evaluation")
        report_lost(comm, current[], err, backtrace())
    end
    try
        while true
            msg = MPI.recv(0, TASK_TAG, comm, nothing)
            msg === nothing && break
            id, x = msg
            current[] = id
            reply = try
                (id, true, f(x...))
            catch e
                (id, false, CapturedException(e, catch_backtrace()))
            end
            try
                MPI.send(reply, 0, RESULT_TAG, comm)
            catch e  # e.g. the value or the exception cannot be serialized
                err = CapturedException(ErrorException("cannot send the result to rank 0: " * sprint(showerror, e)), catch_backtrace())
                MPI.send((id, false, err), 0, RESULT_TAG, comm)
            end
            current[] = 0
        end
    catch e
        active[] && report_lost(comm, current[], e, catch_backtrace())
        rethrow()
    finally
        active[] = false
    end
    return nothing
end

function AdaptivePort.MPIExecutor(body, f; comm = MPI.COMM_WORLD, timeout = Inf)
    MPI.Initialized() || throw(ErrorException("MPIExecutor needs MPI.Init() first"))
    MPI.Comm_size(comm) > 1 || throw(ArgumentError("MPIExecutor needs at least 2 MPI ranks (one manager and one worker); start the script with `mpiexec -n 4 julia ...`"))
    timeout > 0 || throw(ArgumentError("timeout must be positive; got $timeout"))
    MPI.Comm_rank(comm) == 0 || return serve(f, comm)
    s = MPIState(f, comm, timeout)
    poll_task = @async poller(s)
    try
        return body(AdaptivePort.MPIExecutor(s))
    finally
        close!(s)
        wait(poll_task)
    end
end

end
