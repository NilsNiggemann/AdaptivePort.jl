# Run under mpiexec by test_mpi.jl. ARGS[1] is "ok", "throw" or "single"; rank 0 prints what the test checks.
using MPI, AdaptivePort
MPI.Init()
rank = MPI.Comm_rank(MPI.COMM_WORLD)
mode = ARGS[1]

function f(x)
    mode == "throw" && x > 0.4 && error("boom at $x")
    sleep(mode == "abandon" ? 1.0 : 0.02)
    return mode == "ok" ? (x^2, MPI.Comm_rank(MPI.COMM_WORLD)) : x^2
end

if mode == "single"
    try
        MPIExecutor(identity, f)
    catch e
        println("ARGERR ", e isa ArgumentError)
    end
    exit(0)
end

if mode == "abandon"
    try
        MPIExecutor(f) do executor
            executor(f, 0.5)
            executor(f, 0.6)
            executor(f, 0.7)  # queued
            error("body failed")
        end
    catch e
        println("BODYERR $rank $(sprint(showerror, e))")
    end
    exit(0)
end

out = MPIExecutor(f) do executor
    println("NTASKS ", default_ntasks(executor))
    try
        executor(x -> 1, 0.0)
    catch e
        println("DIFFERENT ", e isa ArgumentError)
    end
    if mode == "ok"
        ranks = sort!(unique([fetch(executor(f, x))[2] for x in range(0, 1, 12)]))
        println("RANKS ", ranks)
    else
        learner = Learner1D(f; bounds = (0.0, 1.0))
        try
            BlockingRunner(learner; max_npoints = 100, executor)
        catch e
            println("ERR ", e isa AdaptivePort.EvaluationError, " ", e.exception.ex isa ErrorException, " ", e.point > 0.4)
        end
    end
    "body-value"
end
println("RETURNED $rank $(repr(out))")
