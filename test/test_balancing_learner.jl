@testset "BalancingLearner" begin
    steep(x) = tanh(50x)
    smooth(x) = x  # the loss is scale-free, so only a linear child is clearly easier than tanh(50x) at small n

    function drive!(b, n; batch = 1)
        for _ in 1:(n ÷ batch)
            for p in ask(b, batch)
                tell!(b, p, b.f(p...))
            end
        end
        return b
    end
    counts(b) = [npoints(l) for l in b.learners]

    @testset "routing and f" begin
        b = BalancingLearner([Learner1D(steep; bounds = (-1, 1)), Learner1D(smooth; bounds = (-1, 1))])
        @test b.f(1, 0.3) == steep(0.3) && b.f(2, 0.3) == smooth(0.3)
        @test !any(l -> l === b, b.f.fs)
        pts = ask(b, 4)
        @test all(p -> p[1] in 1:2, pts) && length(pending_points(b)) == 4
        tell!(b, (2, 0.5), 7.0)
        @test npoints(b.learners[2]) == 1 && npoints(b.learners[1]) == 0
        @test result(b)[2].y == [7.0] && isempty(result(b)[1].x)
        @test npoints(b) == 1
        @test_throws ArgumentError BalancingLearner([Learner1D(steep; bounds = (-1, 1)), AverageLearner(steep)])
        @test_throws ArgumentError BalancingLearner(Learner1D[])
        @test_throws ArgumentError BalancingLearner([Learner1D(steep; bounds = (-1, 1))]; strategy = :nope)
        @test_throws ArgumentError BalancingLearner([Learner1D(steep; bounds = (-1, 1))]; cdims = [(a = 1,), (a = 2,)])
        b.strategy = :loss
        @test b.strategy === :loss
        @test_throws ArgumentError (b.strategy = :bogus)
        @test b.strategy === :loss
    end

    @testset "strategies" begin
        mk(strategy) = BalancingLearner([Learner1D(steep; bounds = (-1, 1)), Learner1D(smooth; bounds = (-1, 1))]; strategy)
        b = drive!(mk(:npoints), 40)
        @test abs(counts(b)[1] - counts(b)[2]) <= 1
        b = drive!(mk(:cycle), 40)
        @test counts(b) == [20, 20]
        for s in (:loss, :loss_improvements)
            b = drive!(mk(s), 30)
            c = counts(b)
            @test c[1] > c[2] > 5
            @test c[1] + c[2] == 30
        end
        # a strategy change while running
        b = drive!(mk(:loss_improvements), 20)
        b.strategy = :npoints
        drive!(b, 20)
        @test npoints(b) == 40
        # tie-break: equal losses go to the child with fewer points
        b = BalancingLearner([Learner1D(smooth; bounds = (-1, 1)), Learner1D(smooth; bounds = (-1, 1))]; strategy = :loss)
        drive!(b, 2)
        @test counts(b) == [1, 1]
    end

    @testset "ask with tell_pending = false leaves children untouched" begin
        for strategy in (:loss_improvements, :loss, :npoints, :cycle)
            b = BalancingLearner([Learner1D(steep; bounds = (-1, 1)), Learner1D(smooth; bounds = (-1, 1))]; strategy)
            drive!(b, 6)
            ask(b, 2)  # two pending
            before = pending_points(b)
            cp = b.cycle_pos
            p1 = ask(b, 5; tell_pending = false)
            @test pending_points(b) == before && b.cycle_pos == cp
            @test ask(b, 5; tell_pending = false) == p1
            @test ask(b, 5) == p1
        end
    end

    @testset "children kinds" begin
        f2(x, y) = x^2 + y^2
        b = BalancingLearner([Learner2D(f2; bounds = ((0, 1), (0, 1))), Learner2D((x, y) -> x * y; bounds = ((0, 1), (0, 1)))])
        BlockingRunner(b; max_npoints = 40)
        @test npoints(b) >= 40 && all(c -> c > 3, counts(b))
        p = ask(b, 1)[1]
        @test p[2] isa Tuple

        b = BalancingLearner([AverageLearner(x -> x + 1; atol = 0.1), AverageLearner(x -> x; atol = 0.1)])
        drive!(b, 6)
        @test npoints(b) == 6

        b = BalancingLearner([SequenceLearner(x -> x^2; sequence = 1:5), SequenceLearner(x -> -x; sequence = 1:3)])
        @test AdaptivePort.has_own_goal(b) && !done(b)
        BlockingRunner(b)
        @test done(b) && npoints(b) == 8
        @test result(b)[2].values == [-1, -2, -3]
        @test isempty(ask(b, 3))
        @test_throws ArgumentError done(BalancingLearner([Learner1D(steep; bounds = (-1, 1))]))
    end

    @testset "finished child is skipped" begin
        for strategy in (:loss_improvements, :loss, :npoints, :cycle)
            b = BalancingLearner([SequenceLearner(x -> x, sequence = 1:2), SequenceLearner(x -> x, sequence = 1:6)]; strategy)
            pts = ask(b, 8)
            @test length(pts) == 8 && count(p -> p[1] == 1, pts) == 2
            @test isempty(ask(b, 1))
        end
    end

    @testset "integrator child" begin
        b = BalancingLearner([IntegratorLearner(x -> x^2; bounds = (0, 1), tol = 1e-8), IntegratorLearner(sin; bounds = (0, pi), tol = 1e-8)])
        r = BlockingRunner(b; max_npoints = 500)
        @test done(b)
        @test isapprox(result(b)[1].integral, 1 / 3; atol = 1e-7) && isapprox(result(b)[2].integral, 2; atol = 1e-7)
        # ask without telling pending twice gives the same points
        b = BalancingLearner([IntegratorLearner(x -> x^2; bounds = (0, 1)), IntegratorLearner(sin; bounds = (0, pi))])
        p = ask(b, 6; tell_pending = false)
        @test ask(b, 6; tell_pending = false) == p
        @test isempty(pending_points(b))
        @test length(unique(ask(b, 6))) == 6
    end

    @testset "runner with ntasks > 1" begin
        b = BalancingLearner([Learner1D(steep; bounds = (-1, 1)), Learner1D(smooth; bounds = (-1, 1))])
        BlockingRunner(b; max_npoints = 40, ntasks = 4)
        c = counts(b)
        @test sum(c) >= 40 && c[1] > c[2] > 4
        @test isempty(pending_points(b))
    end

    @testset "steep child gets more points, smooth child not starved" begin
        b = BalancingLearner([Learner1D(steep; bounds = (-1, 1)), Learner1D(smooth; bounds = (-1, 1))])
        drive!(b, 42, batch = 3)
        c = counts(b)
        @test c[1] > c[2] >= 8
    end

    @testset "from_product" begin
        b = AdaptivePort.from_product((x; a, b) -> a * x + b, Learner1D, (bounds = (-1, 1),), (a = 1:2, b = [0.0, 10.0, 20.0]))
        @test length(b.learners) == 6
        @test b.cdims[1] == (a = 1, b = 0.0) && b.cdims[2] == (a = 1, b = 10.0) && b.cdims[4] == (a = 2, b = 0.0)
        @test b.f(4, 0.5) == 1.0 && b.f(2, 0.5) == 10.5
        b = AdaptivePort.from_product((x, y; s) -> s * x * y, Learner2D, Dict(:bounds => ((0, 1), (0, 1))), [:s => [1, 2]])
        @test b.f(2, (0.5, 0.5)) == 0.5 && b.cdims == [(s = 1,), (s = 2,)]
    end

    @testset "tables and files" begin
        mk() = BalancingLearner([Learner1D(steep; bounds = (-1, 1)), Learner1D(smooth; bounds = (-1, 1))])
        b = drive!(mk(), 40)
        t = to_table(b)
        @test first(keys(t)) == :learner_index && length(t.x) == 40 && sort(unique(t.learner_index)) == [1, 2]
        dir = mktempdir()
        for ext in ("csv", "jls")
            path = joinpath(dir, "b.$ext")
            save_data(path, b)
            b2 = load_data!(mk(), path)
            @test npoints(b2) == 40 && counts(b2) == counts(b)
            @test to_table(b2) == t
        end
        @test length(collect(AdaptivePort.table_rows(b, t))) == 40
        # integrator child
        mki() = BalancingLearner([IntegratorLearner(x -> x^2; bounds = (0, 1)), IntegratorLearner(sin; bounds = (0, pi))])
        bi = mki()
        BlockingRunner(bi; max_npoints = 60)
        path = joinpath(dir, "i.jls")
        save_data(path, bi)
        bi2 = load_data!(mki(), path)
        @test npoints(bi2) == npoints(bi)
        # loading clears caches
        b3 = mk()
        ask(b3, 2; tell_pending = false)
        load_data!(b3, joinpath(dir, "b.jls"))
        @test loss(b3) == maximum(loss, b3.learners)
        @test_throws ArgumentError load_data!(BalancingLearner([Learner1D(steep; bounds = (-1, 1))]), joinpath(dir, "b.jls"))
    end

    @testset "nesting-free helpers" begin
        b = BalancingLearner([Learner1D(steep; bounds = (-1, 1))])
        @test isempty(ask(b, 0))
        ask(b, 2)
        remove_unfinished!(b)
        @test isempty(pending_points(b))
        @test sprint(show, b) == "BalancingLearner(1 Learner1D, strategy = :loss_improvements)"
    end
end
