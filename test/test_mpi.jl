using MPI

# Run test/mpi/run_mpi.jl on 3 ranks; returns (finished in time, rank 0 stdout).
function run_mpi(mode; nranks = 3, limit = 300)
    script = joinpath(@__DIR__, "mpi", "run_mpi.jl")
    out = tempname()
    cmd = `$(MPI.mpiexec()) -n $nranks $(Base.julia_cmd()) --project=$(Base.active_project()) $script $mode`
    p = run(pipeline(ignorestatus(cmd); stdout = out, stderr = stderr); wait = false)
    finished = timedwait(() -> process_exited(p), limit) === :ok
    finished || kill(p)
    return finished && success(p), read(out, String)
end

@testset "MPIExecutor" begin
    @testset "results come from the workers" begin
        ok, out = run_mpi("ok")
        @test ok
        @test occursin("NTASKS 2", out)
        @test occursin("RANKS [1, 2]", out)
        @test occursin("DIFFERENT true", out)
        @test occursin("RETURNED 0 \"body-value\"", out)
    end
    @testset "a throwing f is an EvaluationError and every rank exits" begin
        ok, out = run_mpi("throw")
        @test ok
        @test occursin("ERR true true true", out)
        @test occursin("RETURNED 0", out)
    end
    @testset "a body that throws mid-evaluation stops every rank" begin
        ok, out = run_mpi("abandon")
        @test ok
        @test occursin("BODYERR 0 body failed", out)
    end
    @testset "one rank is rejected" begin
        ok, out = run_mpi("single"; nranks = 1)
        @test ok
        @test occursin("ARGERR true", out)
    end
end
