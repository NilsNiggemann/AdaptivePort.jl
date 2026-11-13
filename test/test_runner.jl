using Random

# A function whose evaluations block until released, recording how many run at once.
mutable struct Gate
    const release::Channel{Nothing}
    const started::Threads.Atomic{Int}
    const running::Threads.Atomic{Int}
    const maxrunning::Threads.Atomic{Int}
    Gate() = new(Channel{Nothing}(Inf), Threads.Atomic{Int}(0), Threads.Atomic{Int}(0), Threads.Atomic{Int}(0))
end

function (g::Gate)(x)
    Threads.atomic_add!(g.started, 1)
    n = Threads.atomic_add!(g.running, 1) + 1
    Threads.atomic_max!(g.maxrunning, n)
    take!(g.release)
    Threads.atomic_sub!(g.running, 1)
    return x^2
end

open!(g::Gate, n = 1) = foreach(_ -> put!(g.release, nothing), 1:n)

@testset "Runner" begin
    @testset "BlockingRunner 1D: tol and max_npoints" begin
        l = Learner1D(x -> tanh(50x); bounds = (-1, 1))
        r = BlockingRunner(l; tol = 0.05, max_npoints = 10_000)
        @test status(r) === :finished
        @test done(l; tol = 0.05)
        @test npoints(l) < 10_000

        calls = Threads.Atomic{Int}(0)
        l = Learner1D(x -> (Threads.atomic_add!(calls, 1); tanh(50x)); bounds = (-1, 1))
        BlockingRunner(l; tol = 1e-9, max_npoints = 37, ntasks = 5)
        @test npoints(l) == 37 && calls[] == 37  # hard cap on evaluations, not only on told points
        @test isempty(pending_points(l))
    end

    @testset "BlockingRunner 2D, no duplicate evaluations" begin
        seen = NTuple{2, Float64}[]
        f(x, y) = (push!(seen, (x, y)); exp(-10 * (x^2 + y^2)))
        l = Learner2D(f; bounds = ((-1, 1), (-1, 1)))
        BlockingRunner(l; max_npoints = 50)
        @test npoints(l) == 50
        @test length(seen) == 50 && allunique(seen)
        @test isempty(pending_points(l))

        xs = Float64[]
        lk = ReentrantLock()
        l = Learner1D(x -> (lock(() -> push!(xs, x), lk); sin(8x)); bounds = (-1, 1))
        BlockingRunner(l; max_npoints = 40, ntasks = 3)
        @test length(xs) == 40 && allunique(xs)
    end

    @testset "stops when ask returns nothing" begin
        l = Learner1D(identity; bounds = (1.0, 1.0 + eps()))  # narrower than dx_eps: only the bounds
        r = BlockingRunner(l; tol = 0.0, max_npoints = 100, ntasks = 4)
        @test status(r) === :finished
        @test npoints(l) == 2
    end

    @testset "argument checks" begin
        @test_throws ArgumentError BlockingRunner(Learner1D(identity; bounds = (0, 1)))
        @test_throws ArgumentError BlockingRunner(Learner1D(identity; bounds = (0, 1)); tol = 0.1, ntasks = 0)
        @test_throws ArgumentError BlockingRunner(Learner1D(identity; bounds = (0, 1)); tol = 0.1, checkpoint_interval = 0)
        @test default_ntasks(ThreadedExecutor()) == 1
        @test default_ntasks((f, x) -> nothing) == 1
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
        # the backtrace reaches into the function, not only into the runner
        @test any(((fr, _),) -> occursin("test_runner", string(fr.file)), err.exception.processed_bt)
        @test sprint(showerror, err) isa String
        @test npoints(l) == 1 && isempty(pending_points(l))

        r = AsyncRunner(Learner1D(x -> error("boom"); bounds = (0, 1)); max_npoints = 10)
        @test_throws EvaluationError wait(r)
        @test status(r) === :failed

        # results that arrived before the failure are told
        others = Threads.Atomic{Int}(0)
        function slowfail(x)
            if x == 0.5
                timedwait(() -> others[] == 2, 30.0)
                error("boom")
            end
            Threads.atomic_add!(others, 1)
            return x
        end
        l = Learner1D(slowfail; bounds = (0, 1))
        @test_throws EvaluationError BlockingRunner(l; max_npoints = 3, ntasks = 3)
        @test result(l).x == [0.0, 1.0]
        @test isempty(pending_points(l))
    end

    @testset "refill: a slow point does not hold back new points" begin
        slow = Channel{Nothing}(1)
        f(x) = x == 0 ? (take!(slow); 0.0) : x
        l = Learner1D(f; bounds = (0, 1))
        r = AsyncRunner(l; max_npoints = 20, ntasks = 2)
        # while x = 0 is blocked, the other slot keeps running: all other points pass through it
        @test timedwait(() -> npoints(l) == 19, 30.0) === :ok
        @test status(r) === :running
        put!(slow, nothing)
        wait(r)
        @test npoints(l) == 20
    end

    @testset "ntasks is never exceeded" begin
        g = Gate()
        l = Learner1D(g; bounds = (0, 1))
        r = AsyncRunner(l; max_npoints = 30, ntasks = 3)
        @test timedwait(() -> g.started[] == 3, 30.0) === :ok
        open!(g, 30)
        wait(r)
        @test npoints(l) == 30 && g.started[] == 30
        @test g.maxrunning[] == 3
    end

    @testset "cancel! while evaluations are blocked" begin
        g = Gate()
        l = Learner1D(g; bounds = (0, 1))
        r = AsyncRunner(l; max_npoints = 100, ntasks = 2)
        @test status(r) === :running
        @test timedwait(() -> g.started[] == 2, 30.0) === :ok
        cancel!(r)
        @test wait(r) === r  # returns although both evaluations are still blocked
        @test status(r) === :cancelled
        @test npoints(l) == 0 && isempty(pending_points(l))
        open!(g, 2)  # the abandoned evaluations finish; their results are dropped
        @test timedwait(() -> g.running[] == 0, 30.0) === :ok
        sleep(0.05)
        @test npoints(l) == 0

        r = AsyncRunner(Learner1D(sin; bounds = (0, 1)); tol = 0.01)
        @test status(wait(r)) === :finished
    end

    @testset "duration and end_time return promptly" begin
        g = Gate()
        l = Learner1D(g; bounds = (0, 1))
        t = @elapsed r = BlockingRunner(l; duration = 0.2, ntasks = 2)
        @test status(r) === :finished && t < 10
        @test npoints(l) == 0 && isempty(pending_points(l))
        open!(g, 2)

        l = Learner1D(x -> (sleep(0.01); x); bounds = (0, 1))
        r = BlockingRunner(l; end_time = AdaptivePort.Dates.now() + AdaptivePort.Dates.Millisecond(200))
        @test status(r) === :finished && isempty(pending_points(l))
        r = BlockingRunner(Learner1D(identity; bounds = (0, 1)); duration = AdaptivePort.Dates.Millisecond(50))
        @test status(r) === :finished
    end

    @testset "custom plain-function executor" begin
        used = Threads.Atomic{Int}(0)
        executor = (f, x) -> (Threads.atomic_add!(used, 1); Threads.@spawn f(x...))
        l = Learner2D((x, y) -> x * y; bounds = ((0, 1), (0, 1)))
        BlockingRunner(l; max_npoints = 12, executor, ntasks = 3)
        @test npoints(l) == 12 && used[] == 12

        # a synchronous executor: the value itself is the handle (fetch(x) == x)
        l = Learner1D(sin; bounds = (0, 1))
        BlockingRunner(l; max_npoints = 8, executor = (f, x) -> f(x...))
        @test npoints(l) == 8
    end

    @testset "learners with their own goal stop without tol" begin
        for ntasks in (1, 4)
            l = AverageLearner(seed -> 1 + randn(Random.Xoshiro(seed)); atol = 0.1)
            r = BlockingRunner(l; ntasks)
            @test status(r) === :finished && done(l) && isempty(pending_points(l))
            @test abs(result(l).mean - 1) < 0.5

            l = IntegratorLearner(x -> exp(x); bounds = (0, 1), tol = 1e-10)
            BlockingRunner(l; ntasks)
            # in-flight points may stop at a larger error than tol: upstream's done() does the same (checked against upstream)
            @test done(l) && abs(result(l).integral - (exp(1) - 1)) <= result(l).error
            @test isempty(pending_points(l))

            seq = collect(1:20)
            l = SequenceLearner(x -> x^2; sequence = seq, value_type = Int)
            BlockingRunner(l; ntasks)
            @test done(l) && result(l).values == seq .^ 2
        end
        r = AsyncRunner(SequenceLearner(sqrt; sequence = [1.0, 4.0]))
        @test status(wait(r)) === :finished
        @test result(r.learner).values == [1.0, 2.0]
    end

    @testset "every learner kind stops cleanly on cancel and on failure" begin
        makers = [
            f -> Learner1D(f; bounds = (0, 1)),
            f -> Learner2D((x, y) -> f(x); bounds = ((0, 1), (0, 1))),
            f -> LearnerND((x, y) -> f(x); bounds = ((0, 1), (0, 1))),
            f -> AverageLearner(s -> f(s / 10); atol = 1e-9),
            f -> AverageLearner1D((s, x) -> f(x); bounds = (0, 1)),
            f -> IntegratorLearner(f; bounds = (0, 1), tol = 1e-12),
            f -> SequenceLearner(f; sequence = collect(0:0.1:1), value_type = Float64),
        ]
        for make in makers
            g = Gate()
            l = make(g)
            r = AsyncRunner(l; ntasks = 3)
            @test timedwait(() -> g.started[] == 3, 30.0) === :ok
            cancel!(r)
            wait(r)
            @test status(r) === :cancelled && isempty(pending_points(l))
            open!(g, 3)

            l = make(x -> error("boom"))
            @test_throws EvaluationError BlockingRunner(l; max_npoints = 10, ntasks = 3)
            @test isempty(pending_points(l))
        end
    end

    @testset "vector output and AverageLearner1D under runners" begin
        for ntasks in (1, 4)
            l = Learner1D(x -> (tanh(50x), sin(x)); bounds = (-1, 1), vdim = 2)
            BlockingRunner(l; max_npoints = 60, ntasks)
            @test npoints(l) == 60 && result(l).y isa Vector{NTuple{2, Float64}}

            l = Learner2D((x, y) -> (x * y, exp(-20 * (x^2 + y^2))); bounds = ((-1, 1), (-1, 1)), vdim = 2)
            BlockingRunner(l; max_npoints = 40, ntasks)
            @test npoints(l) == 40 && allunique(result(l).points)

            l = LearnerND((x, y, z) -> tanh(10x) + y * z; bounds = ((-1, 1), (-1, 1), (-1, 1)))
            BlockingRunner(l; max_npoints = 30, ntasks)
            @test npoints(l) == 30 && allunique(result(l).points) && isempty(pending_points(l))

            l = AverageLearner1D((seed, x) -> x^2 + 0.1 * randn(Random.Xoshiro(hash((seed, x)))); bounds = (-1, 1))
            BlockingRunner(l; max_npoints = 15, ntasks)
            @test npoints(l) == 15 && nsamples(l) >= 15 * l.min_samples
        end
        l = AverageLearner1D((seed, x) -> x + 0.01 * randn(Random.Xoshiro(hash((seed, x)))); bounds = (-1, 1))
        @test status(wait(AsyncRunner(l; max_npoints = 10))) === :finished
        l = LearnerND((x, y) -> (x, y^2); bounds = ((0, 1), (0, 1)), vdim = 2)
        @test status(wait(AsyncRunner(l; max_npoints = 20))) === :finished && npoints(l) == 20
    end
end
