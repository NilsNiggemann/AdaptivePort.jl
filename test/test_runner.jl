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

    @testset "learners with their own goal stop without tol" begin
        for parallel in (false, true)
            l = AverageLearner(seed -> 1 + randn(Random.Xoshiro(seed)); atol = 0.1)
            r = BlockingRunner(l; parallel)
            @test status(r) === :finished && done(l) && isempty(pending_points(l))
            @test abs(result(l).mean - 1) < 0.5

            l = IntegratorLearner(x -> exp(x); bounds = (0, 1), tol = 1e-10)
            BlockingRunner(l; parallel)
            # batches (parallel) may stop at a larger error than tol: upstream's done() does the same (checked against upstream)
            @test done(l) && abs(result(l).integral - (exp(1) - 1)) <= result(l).error

            seq = collect(1:20)
            l = SequenceLearner(x -> x^2; sequence = seq, value_type = Int)
            BlockingRunner(l; parallel, batch_size = 3)
            @test done(l) && result(l).values == seq .^ 2
        end
        r = AsyncRunner(SequenceLearner(sqrt; sequence = [1.0, 4.0]))
        @test status(wait(r)) === :finished
        @test result(r.learner).values == [1.0, 2.0]
        @test_throws ArgumentError BlockingRunner(Learner1D(identity; bounds = (0, 1)))
    end

    @testset "vector output and AverageLearner1D under runners" begin
        for parallel in (false, true)
            l = Learner1D(x -> (tanh(50x), sin(x)); bounds = (-1, 1), vdim = 2)
            BlockingRunner(l; max_npoints = 60, parallel, batch_size = 4)
            @test npoints(l) == 60 && result(l).y isa Vector{NTuple{2, Float64}}

            l = Learner2D((x, y) -> (x * y, exp(-20 * (x^2 + y^2))); bounds = ((-1, 1), (-1, 1)), vdim = 2)
            BlockingRunner(l; max_npoints = 40, parallel, batch_size = 4)
            @test npoints(l) == 40 && allunique(result(l).points)

            l = LearnerND((x, y, z) -> tanh(10x) + y * z; bounds = ((-1, 1), (-1, 1), (-1, 1)))
            BlockingRunner(l; max_npoints = 30, parallel, batch_size = 4)
            @test npoints(l) == 30 && allunique(result(l).points) && isempty(pending_points(l))

            l = AverageLearner1D((seed, x) -> x^2 + 0.1 * randn(Random.Xoshiro(hash((seed, x)))); bounds = (-1, 1))
            BlockingRunner(l; max_npoints = 15, parallel)
            @test npoints(l) == 15 && nsamples(l) >= 15 * l.min_samples
        end
        l = AverageLearner1D((seed, x) -> x + 0.01 * randn(Random.Xoshiro(hash((seed, x)))); bounds = (-1, 1))
        @test status(wait(AsyncRunner(l; max_npoints = 10))) === :finished
        l = LearnerND((x, y) -> (x, y^2); bounds = ((0, 1), (0, 1)), vdim = 2)
        @test status(wait(AsyncRunner(l; max_npoints = 20))) === :finished && npoints(l) == 20
    end
end
