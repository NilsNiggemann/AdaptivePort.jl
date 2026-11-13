function run_budget!(l, f, budget, batch)
    while npoints(l) < budget
        pts = ask(l, min(batch, budget - npoints(l)))
        isempty(pts) && break
        for x in pts
            tell!(l, x, f(x))
        end
    end
    return l
end

@testset "Learner1D construction" begin
    l = Learner1D(sin; bounds=(0, 1))
    @test eltype(ask(l, 1)) == Float64
    @test eltype(ask(Learner1D(sin; bounds=(0f0, 1f0)), 1)) == Float32
    for b in ((1, 0), (1, 1), (0, Inf), (NaN, 1), (-Inf, 1))
        @test_throws ArgumentError Learner1D(sin; bounds=b)
    end
    l = Learner1D(sin; bounds=(-1, 1))
    @test npoints(l) == 0
    @test loss(l) == Inf
    @test !done(l; tol=1e6)
    @test isempty(result(l).x)
    @test isempty(result(l).y)
end

@testset "Learner1D sampling" begin
    l = Learner1D(sin; bounds=(-1, 1))
    @test isempty(ask(l, 0))
    @test_throws ArgumentError ask(l, -1)
    @test ask(l, 1) == [-1.0]
    @test ask(l, 1) == [1.0]
    @test ask(Learner1D(sin; bounds=(0, 1)), 3) == [0.0, 1.0, 0.5]

    l = Learner1D(sin; bounds=(-1, 1))
    pts = ask(l, 20)
    @test length(pts) == 20
    @test allunique(pts)
    @test all(-1 .<= pts .<= 1)
    more = ask(l, 20)
    @test isempty(intersect(pts, more))

    l = Learner1D(sin; bounds=(0, 1))
    @test ask(l, 4; tell_pending=false) == ask(l, 4; tell_pending=false)

    l = Learner1D(x -> x^2; bounds=(0, 1))
    tell!(l, 0.0, 0.0)
    tell!(l, 1.0, 1.0)
    @test ask(l, 1) == [0.5]
    tell!(l, 0.5, 0.25)
    x = only(ask(l, 1))
    @test x in (0.25, 0.75)
end

@testset "Learner1D updating" begin
    l = Learner1D(sin; bounds=(0, 1))
    for x in (0.7, 0.2, 0.9)
        tell!(l, x, 10x)
    end
    @test result(l).x == [0.2, 0.7, 0.9]
    @test result(l).y == [2.0, 7.0, 9.0]
    tell!(l, 0.7, -5.0)
    @test npoints(l) == 3
    @test result(l).y == [2.0, 7.0, 9.0]
    @test_throws ArgumentError tell!(l, 2.0, 0.0)
    @test_throws ArgumentError tell!(l, -0.1, 0.0)
    @test_throws ArgumentError tell_pending!(l, 2.0)

    l = Learner1D(x -> x; bounds=(0, 1))
    tell!(l, 0.0, 0.0)
    tell!(l, 1.0, 1.0)
    p = only(ask(l, 1))
    @test p == 0.5
    tell!(l, p, 0.5)
    @test !(p in ask(l, 5))
    @test npoints(l) == 3
    @test loss(l) < Inf
end

@testset "Learner1D adaptive refinement" begin
    f(x) = tanh(50x)
    l = run_budget!(Learner1D(f; bounds=(-1, 1)), f, 100, 1)
    xs = result(l).x
    @test count(x -> abs(x) < 0.1, xs) > 30  # a uniform grid puts ~10 of 100 points there
    @test maximum(diff(xs)) > 10 * minimum(diff(xs))
    @test isfinite(loss(l))

    g(x) = sin(8x) + 0.2cos(23x)
    few = run_budget!(Learner1D(g; bounds=(-1, 1)), g, 10, 1)
    many = run_budget!(Learner1D(g; bounds=(-1, 1)), g, 100, 1)
    @test loss(many) < loss(few)
    @test done(many; tol=loss(many))
    @test !done(many; tol=loss(many) / 2)

    l = Learner1D(x -> 0.0; bounds=(0, 1))
    for x in (0.0, 0.5, 1.0)
        tell!(l, x, 0.0)
    end
    before = loss(l)
    tell!(l, 0.75, 100.0)
    @test loss(l) > before
    @test ask(l, 1)[1] in (0.625, 0.875)

    uniform(xs, ys) = xs[2] - xs[1]
    l = run_budget!(Learner1D(sin; bounds=(0, 1), loss_per_interval=uniform), sin, 17, 1)
    @test result(l).x ≈ range(0, 1; length=17)
end

@testset "Learner1D termination" begin
    l = Learner1D(sign; bounds=(-1, 1))
    for _ in 1:10_000
        done(l; tol=1e-3) && break
        pts = ask(l, 1)
        isempty(pts) && break
        tell!(l, only(pts), sign(only(pts)))
    end
    @test done(l; tol=1e-3) || isempty(ask(l, 1))
    @test npoints(l) < 10_000
end

@testset "Learner1D budgeted batch loop" begin
    f(x) = sin(8x)
    l = run_budget!(Learner1D(f; bounds=(-1, 1)), f, 50, 5)
    xs = result(l).x
    @test npoints(l) == 50
    @test allunique(xs)
    @test all(-1 .<= xs .<= 1)
end
