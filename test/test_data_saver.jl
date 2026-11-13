@testset "DataSaver" begin
    struct Rich
        y::Float64
        note::String
    end
    f1(x) = (y = tanh(50x), err = 0.1x)

    @testset "Learner1D, named tuple, association" begin
        s = DataSaver(Learner1D(f1; bounds = (-1, 1)); arg_picker = r -> r.y)
        @test s.f === s.learner.f
        for x in ask(s, 20)
            tell!(s, x, f1(x))
        end
        ex = extra_data(s)
        @test length(ex) == npoints(s) == 20
        @test all(ex[x] == f1(x) for x in keys(ex))
        r = result(s)
        @test all(r.y .== tanh.(50 .* r.x))
        t = to_table(s)
        @test keys(t) == (:x, :y, :extra_data)
        @test all(t.extra_data[i] == f1(t.x[i]) for i in eachindex(t.x))
        @test t.extra_data[1].y == t.y[1]
    end

    @testset "result types" begin
        for (f, pick) in [
            (x -> (x^2, 1), r -> r[1]),
            (x -> [x^2, 1.0], r -> r[1]),
            (x -> Rich(x^2, "n"), r -> r.y),
        ]
            s = DataSaver(Learner1D(x -> 0.0; bounds = (-1, 1)); arg_picker = pick)
            for x in ask(s, 8)
                tell!(s, x, f(x))
            end
            @test npoints(s) == 8 && length(extra_data(s)) == 8
            @test all(pick(extra_data(s)[x]) == x^2 for x in keys(extra_data(s)))
        end
    end

    @testset "duplicates overwrite" begin
        s = DataSaver(Learner1D(f1; bounds = (-1, 1)); arg_picker = r -> r.y)
        tell!(s, -1.0, (y = 1.0, err = 1))
        tell!(s, -1.0, (y = 2.0, err = 2))
        @test extra_data(s)[-1.0].err == 2
        @test result(s).y == [1.0]  # Learner1D keeps its first value
        @test length(extra_data(s)) == 1
    end

    @testset "Learner2D and AverageLearner" begin
        g(x, y) = (v = x * y, w = "a")
        s = DataSaver(Learner2D(g; bounds = ((-1, 1), (-1, 1))); arg_picker = r -> r.v)
        for p in ask(s, 12)
            tell!(s, p, g(p...))
        end
        @test length(extra_data(s)) == 12 && all(extra_data(s)[p].v == p[1] * p[2] for p in keys(extra_data(s)))
        @test to_table(s).extra_data[3] == extra_data(s)[(to_table(s).x[3], to_table(s).y[3])]

        a = DataSaver(AverageLearner(nothing; atol = 0.1); arg_picker = r -> r.y)
        @test AdaptivePort.has_own_goal(a) == AdaptivePort.has_own_goal(a.learner)
        for sd in ask(a, 5)
            tell!(a, sd, (y = sin(sd), tag = sd))
        end
        @test extra_data(a)[2].tag == 2 && npoints(a) == 5
        @test done(a) == done(a.learner)
    end

    @testset "pending, tell_pending=false" begin
        s = DataSaver(Learner1D(f1; bounds = (-1, 1)); arg_picker = r -> r.y)
        pts = ask(s, 3)
        @test sort(pending_points(s)) == sort(pts)
        remove_unfinished!(s)
        @test isempty(pending_points(s))
        ask(s, 2; tell_pending = false)
        @test isempty(pending_points(s))
    end

    @testset "runner" begin
        s = DataSaver(Learner1D(f1; bounds = (-1, 1)); arg_picker = r -> r.y)
        BlockingRunner(s; max_npoints = 40, ntasks = 3)
        @test npoints(s) >= 40 && length(extra_data(s)) == npoints(s)
        @test all(extra_data(s)[x] == f1(x) for x in keys(extra_data(s)))
    end

    @testset "save and load" begin
        mk() = DataSaver(Learner1D(f1; bounds = (-1, 1)); arg_picker = r -> r.y)
        s = mk()
        BlockingRunner(s; max_npoints = 30)
        mktempdir() do dir
            @test_throws ArgumentError save_data(joinpath(dir, "d.csv"), s)
            p = joinpath(dir, "d.jls")
            save_data(p, s)
            s2 = load_data!(mk(), p)
            @test result(s2) == result(s)
            @test extra_data(s2) == extra_data(s)
            @test to_table(s2) == to_table(s)
            @test length(ask(s2, 3)) == 3
            # an extra column is not numeric but the child alone can go to CSV
            c = joinpath(dir, "c.csv")
            save_data(c, s.learner)
            @test isfile(c)
        end
    end

    @testset "IntegratorLearner child" begin
        mk() = DataSaver(IntegratorLearner(x -> 0.0; bounds = (0, 1), tol = 1e-8); arg_picker = r -> r.v)
        g(x) = (v = exp(x), n = x)
        s = mk()
        for _ in 1:12
            for x in ask(s, 1)
                tell!(s, x, g(x))
            end
        end
        mktempdir() do dir
            p = joinpath(dir, "i.jls")
            save_data(p, s)
            s2 = load_data!(mk(), p)
            @test npoints(s2) == npoints(s)
            @test all(extra_data(s2)[x] == g(x) for x in keys(extra_data(s2)))
            @test length(extra_data(s2)) == npoints(s)
        end
    end
    # not run: balancer of DataSavers (BalancingLearner is not in this worktree)
end
