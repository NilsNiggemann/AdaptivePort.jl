using Distributed

@testset "DistributedExecutor" begin
    pids = addprocs(2)
    try
        @everywhere pids begin
            const CALLS = Int[]
            record(x) = (sleep(0.02); push!(CALLS, myid()); sin(5x))
            failing(x) = x > 0.5 ? throw(DomainError(x, "too big")) : x
        end
        executor = DistributedExecutor(CachingPool(pids))
        @test AdaptivePort.default_ntasks(executor) == 2

        # Every worker gets work; only points and values travel.
        @everywhere pids using AdaptivePort
        learner = Learner1D(x -> Main.record(x); bounds = (-1, 1))
        runner = AsyncRunner(learner; max_npoints = 30, executor)
        @test timedwait(() -> istaskdone(runner.task), 120) == :ok
        wait(runner)
        @test status(runner) === :finished
        counts = [remotecall_fetch(() -> length(Main.CALLS), p) for p in pids]
        @test all(>(0), counts)
        @test sum(counts) >= 30

        # A throwing f becomes an EvaluationError with the point and the original exception.
        l = Learner1D(x -> Main.failing(x); bounds = (0, 1))
        r = AsyncRunner(l; max_npoints = 50, executor)
        @test timedwait(() -> istaskdone(r.task), 120) == :ok
        err = try wait(r); nothing catch e; e end
        @test err isa EvaluationError
        @test err.point > 0.5
        @test err.exception.ex isa DomainError

        # A worker lost mid-run surfaces as an EvaluationError, not a hang.
        @everywhere pids slow(x) = (sleep(60); x)
        l = Learner1D(x -> Main.slow(x); bounds = (0, 1))
        r = AsyncRunner(l; max_npoints = 10, executor = DistributedExecutor(CachingPool(pids[1:1])))
        sleep(2)
        rmprocs(pids[1]; waitfor = 30)
        @test timedwait(() -> istaskdone(r.task), 60) == :ok
        @test (try wait(r); nothing catch e; e end) isa EvaluationError
    finally
        rmprocs(filter(in(procs()), pids); waitfor = 30)
    end
end
