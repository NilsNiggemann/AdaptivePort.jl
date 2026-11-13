module AdaptivePortMPIExt

using AdaptivePort, MPI

const TASK_TAG = 1
const RESULT_TAG = 2

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

# State of the manager (rank 0). Only touched from tasks on the thread that created it, under `lock` for safety.
mutable struct MPIState{F, C}
    const f::F
    const comm::C
    const lock::ReentrantLock
    const idle::Vector{Int}                       # ranks waiting for a point
    const busy::Dict{Int, Int}                    # rank -> id of the point it evaluates
    const queue::Vector{Tuple{Int, Any}}          # points waiting for an idle rank
    const handles::Dict{Int, MPIHandle}
    const requests::Vector{MPI.AbstractRequest}   # sends in flight (they keep their buffers alive)
    nextid::Int
    closed::Bool
end

MPIState(f, comm) = MPIState(f, comm, ReentrantLock(), collect(1:MPI.Comm_size(comm)-1), Dict{Int, Int}(),
                             Tuple{Int, Any}[], Dict{Int, MPIHandle}(), MPI.AbstractRequest[], 0, false)

function post!(s::MPIState, msg, rank, tag)
    push!(s.requests, MPI.isend(msg, rank, tag, s.comm))
    return nothing
end

# Hand queued points to idle ranks.
function dispatch!(s::MPIState)
    while !s.closed && !isempty(s.queue) && !isempty(s.idle)
        rank = popfirst!(s.idle)
        id, x = popfirst!(s.queue)
        s.busy[rank] = id
        post!(s, (id, x), rank, TASK_TAG)
    end
end

# Receive every result that has arrived; returns whether anything came in.
function receive!(s::MPIState)
    got = false
    while true
        isready, status = MPI.Iprobe(MPI.ANY_SOURCE, RESULT_TAG, s.comm, MPI.Status)
        isready || break
        got = true
        rank = status.source
        id, ok, value = MPI.recv(rank, RESULT_TAG, s.comm, nothing)
        delete!(s.busy, rank)
        push!(s.idle, rank)
        h = pop!(s.handles, id, nothing)
        h === nothing || finish!(h, ok, ok ? value : MPIRemoteError(value))
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

AdaptivePort.default_ntasks(e::AdaptivePort.MPIExecutor{<:MPIState}) = MPI.Comm_size(e.state.comm) - 1

# Stop all workers, receive (and drop) the results of evaluations still running, fail every unfinished handle.
function close!(s::MPIState)
    lock(s.lock) do
        s.closed = true
        empty!(s.queue)
        for rank in 1:MPI.Comm_size(s.comm)-1
            post!(s, nothing, rank, TASK_TAG)
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
    return nothing
end

# Worker: serve `f` until the stop message. A failing evaluation is reported and the loop goes on.
function serve(f, comm)
    while true
        msg = MPI.recv(0, TASK_TAG, comm, nothing)
        msg === nothing && return nothing
        id, x = msg
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
    end
end

function AdaptivePort.MPIExecutor(body, f; comm = MPI.COMM_WORLD)
    MPI.Initialized() || throw(ErrorException("MPIExecutor needs MPI.Init() first"))
    MPI.Comm_size(comm) > 1 || throw(ArgumentError("MPIExecutor needs at least 2 MPI ranks (one manager and one worker); start the script with `mpiexec -n 4 julia ...`"))
    MPI.Comm_rank(comm) == 0 || return serve(f, comm)
    s = MPIState(f, comm)
    poll_task = @async poller(s)
    try
        return body(AdaptivePort.MPIExecutor(s))
    finally
        close!(s)
        wait(poll_task)
    end
end

end
