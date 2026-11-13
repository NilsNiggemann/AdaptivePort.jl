@testset "Runner" begin
    @testset "BlockingRunner 1D: tol and max_npoints" begin
        l = Learner1D(x -> tanh(50x); bounds = (-1, 1))
        r = BlockingRunner(l; tol = 0.05, max_npoints = 10_000)
        @test status(r) === :finished
        @test done(l; tol = 0.05)
        @test npoints(l) < 10_000

        l = Learner1D(x -> tanh(50x); bounds = (-1, 1))
        r = BlockingRunner(l; tol = 1e-9, max_npoints = 37, batch_size = 5)
        @test npoints(l) == 37  # hard cap, even though 37 is not a multiple of 5
    end

    @testset "BlockingRunner 2D, no duplicate evaluations" begin
        seen = NTuple{2, Float64}[]
        f(x, y) = (push!(seen, (x, y)); exp(-10 * (x^2 + y^2)))
        l = Learner2D(f; bounds = ((-1, 1), (-1, 1)))
        BlockingRunner(l; max_npoints = 50, batch_size = 4)
        @test npoints(l) == 50
        @test length(seen) == 50 && allunique(seen)
        @test isempty(l.pending)

        xs = Float64[]
        l = Learner1D(x -> (push!(xs, x); sin(8x)); bounds = (-1, 1))
        BlockingRunner(l; max_npoints = 40, batch_size = 3)
        @test length(xs) == 40 && allunique(xs)
    end

    @testset "stops when ask returns nothing" begin
        l = Learner1D(identity; bounds = (1.0, 1.0 + eps()))  # narrower than dx_eps: only the bounds
        r = BlockingRunner(l; tol = 0.0, max_npoints = 100)
        @test status(r) === :finished
        @test npoints(l) == 2
    end

    @testset "BlockingRunner requires a goal" begin
        @test_throws ArgumentError BlockingRunner(Learner1D(identity; bounds = (0, 1)))
        @test_throws ArgumentError BlockingRunner(Learner1D(identity; bounds = (0, 1)); tol = 0.1, batch_size = 0)
    end

    @testset "errors propagate with the failed point" begin
        l = Learner1D(x -> x == 1 ? error("boom") : x; bounds = (0, 1))
        err = try
            BlockingRunner(l; max_npoints = 10)
        catch e
            e
        end
        @test err isa EvaluationError
        @test err.point == 1.0
        @test err.exception.ex == ErrorException("boom")
        @test sprint(showerror, err) isa String
        @test npoints(l) == 1 && isempty(l.pending)

        r = AsyncRunner(Learner1D(x -> error("boom"); bounds = (0, 1)); max_npoints = 10)
        @test_throws EvaluationError wait(r)
        @test status(r) === :failed
    end

    @testset "AsyncRunner returns before finishing; cancel!" begin
        started = Threads.Atomic{Int}(0)
        release = Channel{Nothing}(Inf)
        f(x) = (Threads.atomic_add!(started, 1); take!(release); x^2)
        l = Learner1D(f; bounds = (0, 1))
        r = AsyncRunner(l; max_npoints = 100, batch_size = 2)
        @test status(r) === :running
        @test timedwait(() -> started[] == 1, 30.0) === :ok  # first batch in flight
        @test npoints(l) == 0
        cancel!(r)
        put!(release, nothing)
        put!(release, nothing)
        @test wait(r) === r
        @test status(r) === :cancelled
        @test npoints(l) == 2  # the in-flight batch was told, nothing left pending
        @test isempty(l.pending)

        r = AsyncRunner(Learner1D(sin; bounds = (0, 1)); tol = 0.01)
        @test status(wait(r)) === :finished
    end

    @testset "parallel evaluates a batch concurrently" begin
        n = 4
        started = Threads.Atomic{Int}(0)
        release = Channel{Nothing}(Inf)
        f(x) = (Threads.atomic_add!(started, 1); take!(release); x^2)
        l = Learner1D(f; bounds = (0, 1))
        r = AsyncRunner(l; max_npoints = n, batch_size = n, parallel = true)
        # all n evaluations are in flight at once; sequential evaluation would stall at 1
        @test timedwait(() -> started[] == n, 30.0) === :ok
        foreach(_ -> put!(release, nothing), 1:n)
        wait(r)
        @test npoints(l) == n && allunique(result(l).x)

        # one failure in a parallel batch: the other points are still told
        l = Learner1D(x -> x == 0.5 ? error("boom") : x; bounds = (0, 1))
        @test_throws EvaluationError BlockingRunner(l; max_npoints = 3, batch_size = 3, parallel = true)
        @test result(l).x == [0.0, 1.0]
        @test isempty(l.pending)
    end
end
