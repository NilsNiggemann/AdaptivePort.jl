# Generic property tests, ported from upstream `adaptive/tests/test_learners.py`. Each property runs over the
# learners upstream parametrizes it with; the ones upstream marks `xfail` are noted where they differ.
using Random

@testset "learner properties" begin
    rng = Xoshiro(2024)
    peak(x) = x + 0.3 * tanh(30 * (x - 0.3))  # monotone: the y-range is fixed by the two bounds
    ring(x, y) = x + (hypot(x, y) < 0.6 ? 1.0 : 0.0) + 0.3 * sin(5y)
    ring3(x, y, z) = x + (hypot(x, y, z) < 0.6 ? 1.0 : 0.0)

    kinds = [
        ("Learner1D", () -> Learner1D(peak; bounds = (-1, 1))),
        ("Learner2D", () -> Learner2D(ring; bounds = ((-1, 1), (-1, 1)))),
        ("LearnerND 2D", () -> LearnerND(ring; bounds = ((-1, 1), (-1, 1)))),
        ("LearnerND 3D", () -> LearnerND(ring3; bounds = ntuple(_ -> (-1, 1), 3))),
        ("AverageLearner", () -> AverageLearner(nothing; atol = 0.01)),
        ("AverageLearner1D", () -> AverageLearner1D(nothing; bounds = (-2, 2), min_samples = 3)),
        ("SequenceLearner", () -> SequenceLearner(x -> x[1]^2 + x[2]; sequence = [(a, b) for a in range(-1, 1; length = 20) for b in (0.0, 1.0)])),
        ("IntegratorLearner", () -> IntegratorLearner(exp; bounds = (0, 1), tol = 1e-8)),
    ]
    maker(name) = last(kinds[findfirst(k -> first(k) == name, kinds)])
    # the value of the function at a point of `ask` (AverageLearner and AverageLearner1D have no function)
    value(l, x) = l isa AverageLearner1D ? x[2] + sin(x[2]) + 0.1 * sin(1000 * x[1]) :
        l isa AverageLearner ? sin(x) + 0.1 * sin(1000 * x) :
        
        x isa Tuple ? l.f(x...) : l.f(x)
    value(b::BalancingLearner, x) = value(b.learners[x[1]], x[2])
    tell_all!(l, pts) = foreach(p -> tell!(l, p, value(l, p)), pts)
    # comparable, order-free form of a batch of points (rounded: points agree up to machine precision)
    rnd(x::Real) = round(float(x); digits = 9)
    rnd(x::Tuple) = map(rnd, x)
    rnd(x) = x
    batch(l, n) = sort!(map(rnd, ask(l, n)); by = string)

    @testset "adding existing data is idempotent" begin
        # upstream: Learner1D, Learner2D, LearnerND, SequenceLearner, AverageLearner1D
        for name in ("Learner1D", "Learner2D", "LearnerND 2D", "LearnerND 3D", "SequenceLearner", "AverageLearner1D")
            @testset "$name" begin
                for _ in 1:3
                    l, control = maker(name)(), maker(name)()
                    N = rand(rng, 10:30)
                    ask(control, N)
                    xs = ask(l, N)
                    tell_all!(control, xs)
                    tell_all!(l, xs)
                    tell_all!(l, shuffle(rng, xs))
                    @test npoints(l) == npoints(control)
                    @test loss(l) ≈ loss(control)
                    M = rand(rng, 10:30)
                    @test batch(l, M) == batch(control, M)
                end
            end
        end
    end

    @testset "adding non-chosen data" begin
        # upstream: Learner1D, LearnerND, AverageLearner, AverageLearner1D, SequenceLearner (Learner2D xfail)
        for name in ("Learner1D", "Learner2D", "LearnerND 2D", "LearnerND 3D", "AverageLearner", "AverageLearner1D", "SequenceLearner")
            @testset "$name" begin
                l, control = maker(name)(), maker(name)()
                xs = ask(control, rand(rng, 10:30))
                tell_all!(control, xs)
                tell_all!(l, xs)  # `l` never saw `ask`
                M = rand(rng, 10:30)
                @test batch(l, M) == batch(control, M)
            end
        end
    end

    @testset "point order is irrelevant" begin
        # upstream: Learner1D, AverageLearner, AverageLearner1D (Learner2D and LearnerND xfail, left out)
        # Learner1D: the lazy y-scale (as upstream's default `_recompute_losses_factor = 2`, which upstream's test sets to 1)
        # makes the order matter when the y-range grows; so the bounds, which fix the range of `peak`, are told first.
        for name in ("Learner1D", "AverageLearner", "AverageLearner1D")
            @testset "$name" begin
                for _ in 1:3
                    l, control = maker(name)(), maker(name)()
                    N = rand(rng, 10:30)
                    ask(control, N)
                    xs = ask(l, N)
                    tell_all!(control, xs)
                    if name == "Learner1D"
                        ends = filter(in((-1.0, 1.0)), xs)
                        tell_all!(l, ends)
                        tell_all!(l, shuffle(rng, setdiff(xs, ends)))
                    else
                        tell_all!(l, shuffle(rng, xs))
                    end
                    @test loss(l) ≈ loss(control)
                    if name in ("Learner1D", "AverageLearner", "AverageLearner1D")
                        @test batch(l, 20) == batch(control, 20)
                    end
                end
            end
        end
    end

    @testset "expected loss improvement is below the total loss" begin
        # upstream: Learner1D, LearnerND, AverageLearner, AverageLearner1D (Learner2D xfail)
        for name in ("Learner1D", "Learner2D", "LearnerND 2D", "LearnerND 3D", "AverageLearner", "AverageLearner1D")
            @testset "$name" begin
                l = maker(name)()
                for _ in 1:2
                    tell_all!(l, ask(l, rand(rng, name == "LearnerND 3D" ? (20:40) : (50:100))))
                end
                _, improvements = AdaptivePort.ask_with_improvements(l, rand(rng, 50:100))
                @test all(>=(0), improvements)
                @test l isa AverageLearner ? sum(improvements) < loss(l) : true
                @test maximum(improvements) <= loss(l) * (1 + 1e-6)
            end
        end
    end

    @testset "performance is invariant under scaling of x and y" begin
        # upstream: Learner1D, LearnerND, AverageLearner1D (Learner2D xfail)
        for name in ("Learner1D", "LearnerND")
            @testset "$name" begin
                xscale, yscale = 1000 * rand(rng), 1000 * rand(rng)
                g(x...) = name == "Learner1D" ? sin(3 * x[1] + 0.2) + 0.5 * tanh(20 * (x[1] - 0.1)) : x[1]^2 + tanh(5 * (x[1] + x[2]))
                mk(f, s) = name == "Learner1D" ? Learner1D(f; bounds = (-s, s)) : LearnerND(f; bounds = ((-s, s), (-s, s)))
                control = mk(g, 1.0)
                scaled = mk((x...) -> yscale * g((x ./ xscale)...), xscale)
                for _ in 1:(name == "Learner1D" ? 200 : 30)
                    cx = only(ask(control, 1))
                    sx = only(ask(scaled, 1))
                    @test collect(sx ./ xscale) ≈ collect(cx) atol = 1e-9
                    tell!(control, cx, control.f(cx...))
                    tell!(scaled, sx, scaled.f(sx...))
                end
                @test loss(scaled) ≈ loss(control) rtol = 1e-9
            end
        end
        @testset "AverageLearner1D" begin
            xscale, yscale = 1000 * rand(rng), 1000 * rand(rng)
            control = AverageLearner1D(nothing; bounds = (-2, 2), min_samples = 3)
            scaled = AverageLearner1D(nothing; bounds = (-2xscale, 2xscale), min_samples = 3)
            for _ in 1:150
                (cs, cx) = only(ask(control, 1))
                (ss, sx) = only(ask(scaled, 1))
                @test cs == ss
                @test sx / xscale ≈ cx atol = 1e-9
                tell!(control, (cs, cx), tanh(3cx))
                tell!(scaled, (ss, sx), yscale * tanh(3sx / xscale))
            end
            @test loss(scaled) ≈ loss(control) rtol = 1e-9
        end
    end

    @testset "ask with tell_pending = false does not change the learner" begin
        for (name, _) in kinds
            @testset "$name" begin
                l = maker(name)()
                tell_all!(l, ask(l, 12))
                before = (loss(l), npoints(l), pending_points(l), to_table(l))
                p1 = ask(l, 5; tell_pending = false)
                @test (loss(l), npoints(l), pending_points(l), to_table(l)) == before
                @test ask(l, 5; tell_pending = false) == p1
                @test ask(l, 5) == p1  # the same points come out again, and now they are pending
                @test length(pending_points(l)) == length(before[3]) + length(p1)
            end
        end
    end

    @testset "balancing learner with pending and late points" begin
        # upstream: Learner1D, Learner2D, LearnerND, AverageLearner, AverageLearner1D, SequenceLearner
        for name in ("Learner1D", "Learner2D", "LearnerND 2D", "AverageLearner", "AverageLearner1D", "SequenceLearner")
            @testset "$name" begin
                b = BalancingLearner([maker(name)() for _ in 1:4])
                stash = Any[]
                for _ in 1:60
                    n = rand(rng, 1:10)
                    m = rand(rng, 0:n)
                    xs = shuffle(rng, ask(b, n; tell_pending = false))
                    for _ in 1:min(m, length(xs))
                        push!(stash, pop!(xs))
                    end
                    foreach(x -> tell!(b, x, value(b, x)), xs)
                    shuffle!(rng, stash)
                    for _ in 1:min(m, length(stash))
                        x = pop!(stash)
                        tell!(b, x, value(b, x))
                    end
                end
                @test all(l -> (l isa AverageLearner1D ? nsamples(l) : npoints(l)) > 5, b.learners)
            end
        end
    end

    # upstream's `simple_run`: serial ask/tell up to `n` points, stopping at the learner's own goal
    function simple_run!(l, n)
        for _ in 1:n
            has_own_goal(l) && done(l) && break
            tell_all!(l, ask(l, 1))
        end
        return l
    end
    has_own_goal(l) = AdaptivePort.has_own_goal(l)
    has_own_goal(b::BalancingLearner) = false
    has_own_goal(s::DataSaver) = AdaptivePort.has_own_goal(s.learner)
    count_samples(l) = l isa AverageLearner1D ? nsamples(l) : npoints(l)
    count_samples(b::BalancingLearner) = sum(count_samples, b.learners)
    count_samples(s::DataSaver) = count_samples(s.learner)
    lazy_scale(l) = l isa Union{Learner1D,AverageLearner1D} || (l isa BalancingLearner && any(lazy_scale, l.learners)) ||
                    (l isa DataSaver && lazy_scale(l.learner))
    # `Learner1D` and `AverageLearner1D` rescale losses lazily (as upstream, whose test sets `_recompute_losses_factor = 1`),
    # so a reload agrees on the data and on the loss only up to that factor of 2 in the y-scale
    function same_loss(a, b)
        lazy_scale(a) ? isapprox(loss(a), loss(b); rtol = 0.5) : isapprox(loss(a), loss(b))
    end
    # a reloaded learner can sample on; a finished one (own goal reached, or sequence exhausted) has nothing to ask
    function keeps_running(l, before)
        finished = has_own_goal(l) && done(l)
        finished || simple_run!(l, 20)
        return finished || count_samples(l) > before
    end
    ext_for(name) = name == "SequenceLearner" ? ("jls",) : ("csv", "jls")

    @testset "saving round trips" begin
        # upstream: all learners; a balancer of one; DataSaver of all but SequenceLearner
        for (name, mk) in kinds, ext in ext_for(name)
            @testset "$name .$ext" begin
                l, control = mk(), mk()
                simple_run!(l, 100)
                mktempdir() do dir
                    load_data!(control, save_data(joinpath(dir, "data.$ext"), l))
                    @test same_loss(control, l)
                    @test npoints(control) == npoints(l)
                    @test keeps_running(control, count_samples(control))
                end
            end
        end
        @testset "BalancingLearner $name" for (name, mk) in kinds
            b, control = BalancingLearner([mk()]), BalancingLearner([mk()])
            for _ in 1:100
                foreach(x -> tell!(b, x, value(b, x)), ask(b, 1))
            end
            mktempdir() do dir
                load_data!(control, save_data(joinpath(dir, "bal.jls"), b))
                @test same_loss(control, b)
                @test npoints(control) == npoints(b)
                n = count_samples(control)
                for _ in 1:20
                    foreach(x -> tell!(control, x, value(control, x)), ask(control, 1))
                end
                name in ("SequenceLearner", "IntegratorLearner") || @test count_samples(control) > n
            end
        end
        @testset "DataSaver $name" for (name, mk) in kinds
            name == "SequenceLearner" && continue
            s, control = DataSaver(mk(); arg_picker = r -> r.y), DataSaver(mk(); arg_picker = r -> r.y)
            for _ in 1:100
                has_own_goal(s) && done(s) && break
                foreach(x -> tell!(s, x, (y = value(s.learner, x), t = hash(x))), ask(s, 1))
            end
            mktempdir() do dir
                load_data!(control, save_data(joinpath(dir, "ds.jls"), s))
                @test same_loss(control, s)
                @test extra_data(control) == extra_data(s)
            end
        end
    end
end
