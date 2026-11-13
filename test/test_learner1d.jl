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

@testset "Learner1D vector output" begin
    f(x) = (tanh(50x), sin(x))
    learn(g; n=20, kw...) = begin
        l = Learner1D(g; bounds=(-1, 1), kw...)
        for _ in 1:n
            x = only(ask(l, 1))
            tell!(l, x, g(x))
        end
        l
    end

    # Reference: upstream adaptive, Learner1D(f; bounds=(-1, 1)), 20 points asked one at a time
    # (`uv run --no-project --with adaptive python`); losses are upstream's `learner.losses`. Tolerance 1e-12.
    xs_ref = [-1.0, -0.75, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625, -0.0078125, -0.00390625, 0.0,
              0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0]
    losses_ref = [0.1483630176961082, 0.16077172241776735, 0.17053886161511683, 0.08758918723574916,
                  0.04409367282039791, 0.04309345662639039, 0.13143287483455354, 0.14081652916970835,
                  0.08953761348360394, 0.09645292321003168, 0.18599047988245615, 0.14081652916970835,
                  0.13143287483455354, 0.04309345662639039, 0.04409367282039791, 0.08758918723574916,
                  0.17053886161511683, 0.16077172241776735, 0.1483630176961082]
    l = learn(f; vdim=2)
    @test result(l).x == xs_ref
    @test result(l).y isa Vector{NTuple{2,Float64}}
    @test isapprox(l.losses, losses_ref; rtol=1e-12)
    @test isapprox(loss(l), 0.18599047988245615; rtol=1e-12)

    # refinement concentrates near the steep component's jump at 0
    xs = result(l).x
    @test count(x -> abs(x) < 0.1, xs) > count(x -> abs(x) > 0.5, xs)

    # a common factor on all components leaves the point sequence unchanged
    l2 = learn(x -> 1000 .* f(x); vdim=2)
    @test result(l2).x == xs

    # vectors, tuples and wrong lengths
    l = Learner1D(f; bounds=(0, 1), vdim=3)
    tell!(l, 0.0, [1, 2, 3])
    tell!(l, 1.0, (4.0, 5.0, 6.0))
    @test result(l).y == [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
    @test_throws ArgumentError tell!(l, 0.5, [1.0, 2.0])
    @test_throws ArgumentError tell!(l, 0.5, (1.0, 2.0, 3.0, 4.0))
    @test_throws ArgumentError Learner1D(f; bounds=(0, 1), vdim=0)
    @test npoints(l) == 2

    # default_loss: maximum over components
    @test AdaptivePort.default_loss((0.0, 0.3), ((0.0, 0.0), (0.4, 0.1))) ≈ 0.5
end

@testset "Learner1D update_value!" begin
    l = Learner1D(sin; bounds=(0, 1))
    @test_throws ArgumentError AdaptivePort.update_value!(l, 0.5, 1.0)
    tell!(l, 0.0, 0.0)
    tell!(l, 0.5, 0.5)
    tell!(l, 1.0, 1.0)
    old = copy(l.losses)
    AdaptivePort.update_value!(l, 0.5, 0.0)
    @test result(l).y == [0.0, 0.0, 1.0]
    @test l.losses[1] < old[1]
    @test l.losses[2] > old[2]
    @test_throws ArgumentError AdaptivePort.update_value!(l, 0.25, 0.0)

    # vector output; the range does not shrink (upstream), a large new value rescales everything
    l = Learner1D(sin; bounds=(0, 1), vdim=2)
    tell!(l, 0.0, (0.0, 0.0))
    tell!(l, 1.0, (1.0, 0.0))
    AdaptivePort.update_value!(l, 1.0, [0.0, 10.0])
    @test result(l).y == [(0.0, 0.0), (0.0, 10.0)]
    @test l.yscale == 10.0
    @test l.losses[1] ≈ hypot(1.0, 1.0)
    @test_throws ArgumentError AdaptivePort.update_value!(l, 1.0, [1.0])
end

@testset "Learner1D loss(real = false) and ask_with_improvements" begin
    for (f, vdim) in ((x -> tanh(50x), 1), (x -> (tanh(50x), x^2), 2))
        l = Learner1D(f; bounds = (-1, 1), vdim)
        @test loss(l; real = false) == Inf
        pts, imp = AdaptivePort.ask_with_improvements(l, 2)
        @test pts == [-1.0, 1.0] && imp == [Inf, Inf]
        @test loss(l; real = false) == Inf  # bounds pending, no evaluated neighbours
        foreach(x -> tell!(l, x, f(x)), pts)
        pending_before = copy(l.pending)
        for _ in 1:20
            before = loss(l; real = false)
            @test before <= loss(l)
            (p,), (i,) = AdaptivePort.ask_with_improvements(l, 1; tell_pending = false)
            @test l.pending == pending_before
            @test i ≈ before / 2  # the largest priority is halved by the split
            @test ask(l, 1; tell_pending = false) == [p]
            tell_pending!(l, p)
            @test loss(l; real = false) <= before
            if length(l.pending) > 3
                x = first(l.pending)
                tell!(l, x, f(x))
            end
            pending_before = copy(l.pending)
        end
    end
    # a pending point outside the evaluated range is an interval with infinite loss
    l = Learner1D(zero; bounds = (0, 1))
    tell!(l, 0.0, 0.0)
    tell!(l, 0.01, 0.0)
    tell_pending!(l, 1.0)
    @test loss(l; real = false) == Inf
    @test loss(l) == Inf
    (p,), (i,) = AdaptivePort.ask_with_improvements(l, 1)
    @test p ≈ 0.505 && i == Inf
    # a pending point splits an interval's loss proportionally
    l = Learner1D(x -> x^2; bounds = (0, 1))
    foreach(x -> tell!(l, x, x^2), (0.0, 1.0))
    L = loss(l)
    tell_pending!(l, 0.25)
    @test loss(l; real = false) ≈ 0.75 * L
    @test loss(l) == L
    @test ask(l, 3) isa Vector{Float64}
end
