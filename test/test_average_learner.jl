using Random

@testset "AverageLearner" begin
    f_unused(seed) = error("this function shouldn't be used")

    @testset "only returns new points" begin
        l = AverageLearner(f_unused; rtol = 0.01)
        for i in 5:9
            tell!(l, i, 1.0)
        end
        tell_pending!(l, 0)  # so 0 is not returned anymore
        @test [only(ask(l, 1)) for _ in 1:5] == [1, 2, 3, 4, 10]
        @test pending_points(l) == [0, 1, 2, 3, 4, 10]
        remove_unfinished!(l)
        @test isempty(pending_points(l))
        @test ask(l, 3; tell_pending = false) == [0, 1, 2]
        @test ask(l, 3; tell_pending = false) == [0, 1, 2]
        @test ask(l, 0) == Int[]
    end

    @testset "mean, std and npoints match a direct computation" begin
        l = AverageLearner(f_unused; rtol = 0.01)
        rng = Random.Xoshiro(1)
        told = Dict{Int,Float64}()
        for i in 1:300
            n = rand(rng, 0:600)
            value = rand(rng)
            if value < 0.9
                tell!(l, n, value)
                get!(told, n, value)  # a repeated seed keeps its first value
            else
                tell_pending!(l, n)
            end
            if i > 2 && i % 10 == 0
                vals = collect(values(told))
                r = result(l)
                @test npoints(l) == length(told) == r.n
                @test r.mean ≈ sum(vals) / length(vals) atol = 1e-12
                @test r.std ≈ sqrt(sum(abs2, vals .- sum(vals) / length(vals)) / (length(vals) - 1)) atol = 1e-12
            end
        end
    end

    @testset "min_npoints" begin
        for min_npoints in 1:3
            l = AverageLearner(seed -> 0.1; atol = 0.01, rtol = 0.01, min_npoints)
            BlockingRunner(l; tol = 1.0)
            @test npoints(l) >= max(2, min_npoints)
        end
        l = AverageLearner(f_unused; atol = 1.0, min_npoints = 3)
        tell!(l, 0, 1.0)
        tell!(l, 1, 2.0)
        @test loss(l) == Inf && result(l).std == Inf && !done(l)
        tell!(l, 2, 3.0)
        @test isfinite(loss(l))
    end

    @testset "zero mean" begin  # upstream issue 275
        l = AverageLearner(f_unused; rtol = 0.01)
        tell!(l, 0, -1.0)
        tell!(l, 1, 1.0)
        @test isfinite(loss(l))
    end

    @testset "requires a tolerance" begin
        @test_throws ArgumentError AverageLearner(f_unused)
    end

    @testset "loss agrees with upstream" begin
        # Reference from python-adaptive (AverageLearner), values 0.3 1.7 -0.4 2.2 0.9 1.1 0.05 told, seeds 7 and 8 pending.
        # Compared with rtol 1e-12.
        vals = [0.3, 1.7, -0.4, 2.2, 0.9, 1.1, 0.05]
        refs = [
            (; atol = 0.1) => (3.489293634537972, 3.07726773608939, 0.20601294922429103),
            (; rtol = 0.1) => (4.1752231524385985, 3.682200709850552, 0.24651122129402325),
            (; atol = 0.5, rtol = 0.2) => (2.0876115762192993, 1.841100354925276, 0.12325561064701163),
        ]
        for (kw, (loss_real, loss_all, improvement)) in refs
            l = AverageLearner(f_unused; kw...)
            foreach(((i, v),) -> tell!(l, i - 1, v), enumerate(vals))
            tell_pending!(l, 7)
            tell_pending!(l, 8)
            r = result(l)
            @test r.mean ≈ 0.8357142857142857 rtol = 1e-12
            @test r.std ≈ 0.9231803208268171 rtol = 1e-12
            @test loss(l) ≈ loss_real rtol = 1e-12
            @test loss(l; real = false) ≈ loss_all rtol = 1e-12
            pts, imps = AdaptivePort.ask_with_improvements(l, 2; tell_pending = false)
            @test pts == [9, 10] && all(≈(improvement; rtol = 1e-12), imps)
        end
        l = AverageLearner(f_unused; atol = 1.0)
        @test AdaptivePort.ask_with_improvements(l, 2)[2] == [Inf, Inf]
    end

    @testset "BlockingRunner stops on its own" begin
        l = AverageLearner(seed -> randn(Random.Xoshiro(seed)); atol = 0.1)
        r = BlockingRunner(l)
        @test status(r) === :finished
        @test done(l) && loss(l) <= 1
        @test abs(result(l).mean) < 0.5
        @test npoints(l) < 1000
    end
end

@testset "AverageLearner loss with only pending points" begin
    l = AverageLearner(identity; atol = 0.1)
    ask(l, 3)
    @test loss(l; real = false) == Inf
    tell!(l, 0, 1.0)
    @test loss(l; real = false) == Inf
end
