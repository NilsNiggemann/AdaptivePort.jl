function run_budget2!(l, f, budget, batch)
    while npoints(l) < budget
        pts = ask(l, min(batch, budget - npoints(l)))
        isempty(pts) && break
        for p in pts
            tell!(l, p, f(p...))
        end
    end
    return l
end

signed_area(pts, (i, j, k)) = ((pts[j][1] - pts[i][1]) * (pts[k][2] - pts[i][2]) - (pts[k][1] - pts[i][1]) * (pts[j][2] - pts[i][2])) / 2

total_area(l) = sum(t -> signed_area(result(l).points, t), triangles(l); init=0.0)

@testset "Learner2D construction" begin
    l = Learner2D((x, y) -> x + y; bounds=((-1, 1), (-1, 1)))
    @test npoints(l) == 0
    @test loss(l) == Inf
    @test !done(l; tol=1e6)
    @test isempty(result(l).points)
    @test isempty(result(l).values)
    @test isempty(triangles(l))
    @test eltype(ask(Learner2D(+; bounds=((0, 2), (0, 1))), 1)) == NTuple{2,Float64}
    @test eltype(ask(Learner2D(+; bounds=((0f0, 2f0), (0f0, 1f0))), 1)) == NTuple{2,Float32}
    for b in (((1, 0), (0, 1)), ((0, 1), (1, 1)), ((0, Inf), (0, 1)), ((0, 1), (NaN, 1)), ((0, 1), (-Inf, 1)))
        @test_throws ArgumentError Learner2D(+; bounds=b)
    end
end

@testset "Learner2D initial sampling" begin
    corners = [(0.0, 0.0), (0.0, 2.0), (1.0, 0.0), (1.0, 2.0)]
    l = Learner2D(+; bounds=((0, 1), (0, 2)))
    @test isempty(ask(l, 0))
    @test_throws ArgumentError ask(l, -1)
    @test ask(l, 4) == corners
    l = Learner2D(+; bounds=((0, 1), (0, 2)))
    @test [only(ask(l, 1)) for _ in 1:4] == corners
end

@testset "Learner2D point selection" begin
    f(x, y) = tanh(50x) + y
    l = Learner2D(f; bounds=((-1, 1), (-1, 1)))
    run_budget2!(l, f, 40, 5)
    pts = ask(l, 50)
    @test allunique(pts)
    @test all(p -> -1 <= p[1] <= 1 && -1 <= p[2] <= 1, pts)
    @test isempty(intersect(pts, result(l).points))
    more = ask(l, 50)
    @test isempty(intersect(pts, more))

    l = Learner2D(f; bounds=((-1, 1), (-1, 1)))
    run_budget2!(l, f, 40, 5)
    @test ask(l, 10; tell_pending=false) == ask(l, 10; tell_pending=false)
end

@testset "Learner2D updating" begin
    l = Learner2D(+; bounds=((0, 1), (0, 1)))
    tell!(l, (0.5, 0.5), 3.0)
    @test npoints(l) == 1
    @test result(l).points == [(0.5, 0.5)]
    @test result(l).values == [3.0]
    tell!(l, (0.5, 0.5), 4.0)
    @test npoints(l) == 1
    @test result(l).values == [4.0]
    @test_throws ArgumentError tell!(l, (2.0, 0.5), 1.0)
    @test_throws ArgumentError tell!(l, (0.5, -0.1), 1.0)
    @test_throws ArgumentError tell_pending!(l, (2.0, 0.5))
    tell_pending!(l, (0.5, 0.5))
    @test npoints(l) == 1
    tell_pending!(l, (0.25, 0.25))
    @test (0.25, 0.25) ∉ ask(l, 20)
end

@testset "Learner2D triangulation" begin
    f(x, y) = x * y
    l = Learner2D(f; bounds=((-1, 3), (0, 1)))
    for p in ask(l, 4)
        tell!(l, p, f(p...))
    end
    @test length(triangles(l)) == 2
    @test total_area(l) ≈ 4
    run_budget2!(l, f, 100, 5)
    tri = triangles(l)
    @test length(tri) > 2
    @test total_area(l) ≈ 4
    pts = result(l).points
    @test all(t -> all(i -> 1 <= i <= length(pts), t), tri)
    @test all(t -> signed_area(pts, t) > 0, tri)

    l = Learner2D(f; bounds=((0, 1), (0, 1)))
    for x in 0:0.25:1
        tell!(l, (x, x), x)
    end
    @test isempty(triangles(l))
    @test loss(l) == Inf
end

@testset "Learner2D adaptivity" begin
    f(x, y) = tanh(50x)
    bounds = ((-1, 1), (-1, 1))
    for batch in (1, 5)
        l = Learner2D(f; bounds)
        run_budget2!(l, f, 20, batch)
        l20 = loss(l)
        run_budget2!(l, f, 300, batch)
        @test npoints(l) == 300
        @test isfinite(loss(l))
        @test loss(l) < l20
        @test done(l; tol=loss(l))
        near = count(p -> abs(p[1]) < 0.1, result(l).points) / 300
        @test near > 0.25
    end

    # loss is invariant under rescaling of x and of z
    la = Learner2D(f; bounds)
    lb = Learner2D((x, y) -> 100f(x / 10, y); bounds=((-10, 10), (-1, 1)))
    for p in ask(la, 4)
        tell!(la, p, f(p...))
    end
    for p in ask(lb, 4)
        tell!(lb, p, 100f(p[1] / 10, p[2]))
    end
    for _ in 1:30
        for (l, g) in ((la, f), (lb, (x, y) -> 100f(x / 10, y)))
            p = only(ask(l, 1))
            tell!(l, p, g(p...))
        end
    end
    @test collect(Iterators.flatten(result(lb).points)) ≈ collect(Iterators.flatten([(10x, y) for (x, y) in result(la).points]))
    @test loss(la) ≈ loss(lb) rtol=1e-6

    # flat function
    l = run_budget2!(Learner2D((x, y) -> 0.0; bounds), (x, y) -> 0.0, 100, 5)
    @test npoints(l) > 4
    @test isfinite(loss(l))
    @test !isnan(loss(l))

    # NaN values do not crash
    g(x, y) = x > 0.5 ? NaN : x + y
    l = run_budget2!(Learner2D(g; bounds), g, 100, 5)
    @test npoints(l) > 4
    @test !isnan(loss(l))
end

@testset "Learner2D termination" begin
    f(x, y) = sin(x) * cos(y)
    l = Learner2D(f; bounds=((-1, 1), (-1, 1)))
    n = 0
    while !done(l; tol=0.05) && n < 5000
        pts = ask(l, 5)
        isempty(pts) && break
        for p in pts
            tell!(l, p, f(p...))
        end
        n += 1
    end
    @test done(l; tol=0.05)
    @test n < 5000
end

@testset "Learner2D loss matches scipy/upstream" begin
    # Reference gradients from scipy's CloughTocher2DInterpolator(Delaunay(p), v, tol=1e-6).grad,
    # which upstream's deviations() uses.
    p = [(-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5), (0.1, 0.2), (-0.2, -0.1), (0.3, -0.3), (-0.25, 0.35)]
    v = [sin(3x) * cos(2y) + y^2 for (x, y) in p]
    ref = [(0.9745733157553469, -1.5946806354965544), (0.08671324629315874, 1.25156107268391), (0.6851106098109432, 0.7639642623115477), (1.465972167188699, -0.6390303345631257), (2.4452873022258697, -0.12036383658612544), (2.0999547712898257, -0.03657882986767205), (1.7077021282618345, 0.2605390848047694), (1.2827119772120799, 1.0454235934246405)]
    ts = AdaptivePort.delaunay_triangles(p)
    g = AdaptivePort.estimate_gradients(p, v, ts)
    @test all(i -> all(isapprox.(g[i], ref[i]; atol=1e-10)), eachindex(p))
    @test length(default_loss(p, v, ts)) == length(ts)
    @test all(>(0), default_loss(p, v, ts))
end

@testset "Learner2D vector output" begin
    # localised ring in one component, smooth plane in the other: sampling concentrates on the ring
    f(x, y) = (exp(-50 * (x^2 + y^2 - 0.3)^2), (x + y) / 10)
    l = Learner2D(f; bounds=((-1, 1), (-1, 1)), vdim=2)
    run_budget2!(l, f, 300, 5)
    @test npoints(l) == 300
    @test eltype(result(l).values) == NTuple{2,Float64}
    @test isfinite(loss(l))
    on_ring = count(p -> abs(p[1]^2 + p[2]^2 - 0.3) < 0.1, result(l).points) / 300
    @test on_ring > 0.3  # the annulus holds 16% of the area (measured 0.34)

    # tell! accepts any iterable of length vdim and rejects other lengths
    l = Learner2D(f; bounds=((0, 1), (0, 1)), vdim=3)
    tell!(l, (0.5, 0.5), [1, 2, 3])
    tell!(l, (0.25, 0.5), (1, 2, 3.5))
    @test result(l).values == [(1.0, 2.0, 3.0), (1.0, 2.0, 3.5)]
    @test_throws ArgumentError tell!(l, (0.1, 0.1), (1.0, 2.0))
    @test_throws ArgumentError tell!(l, (0.1, 0.1), 1.0)
    @test_throws ArgumentError Learner2D(f; bounds=((0, 1), (0, 1)), vdim=0)
    @test npoints(l) == 2

    # a value with any non-finite component is left out of the triangulation
    l = Learner2D(f; bounds=((0, 1), (0, 1)), vdim=2)
    for (p, z) in zip(ask(l, 4), ((0, 0), (0, 1), (1, 0), (1, 1)))
        tell!(l, p, z)
    end
    tell!(l, (0.5, 0.5), (1.0, NaN))
    @test npoints(l) == 5
    @test length(triangles(l)) == 2
    @test all(t -> 5 ∉ t, triangles(l))
    g(x, y) = (x + y, x > 0.5 ? NaN : x * y)
    l = run_budget2!(Learner2D(g; bounds=((0, 1), (0, 1)), vdim=2), g, 100, 5)
    @test npoints(l) > 4
    @test !isnan(loss(l))
end

@testset "Learner2D vector loss matches scipy/upstream" begin
    # Produced with upstream's default_loss(LinearNDInterpolator(p, v)) (adaptive + scipy, uv run); losses are keyed by
    # the sorted 1-based vertex triple. Gradients: CloughTocher2DInterpolator(tri, v / ptp(v, axis=0).max(), tol=1e-6).grad
    # for the second component.
    p = [(-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5), (0.1, 0.2), (-0.2, -0.1), (0.3, -0.3), (-0.25, 0.35)]
    v = [(sin(3x) * cos(2y) + y^2, 5exp(-50 * (x^2 + y^2 - 0.1)^2), x * y) for (x, y) in p]
    ref = Dict((1, 2, 6) => 1.8307819515911237, (1, 3, 7) => 1.8489064898499101, (1, 6, 7) => 1.319572105994419, (5, 6, 7) => 0.713339252482134, (3, 4, 7) => 1.777344171832272,
               (4, 5, 7) => 1.131790485454905, (2, 6, 8) => 0.5467508977571149, (5, 6, 8) => 0.4617089607595119, (2, 4, 8) => 1.5838310679482115, (4, 5, 8) => 0.8283910187751977)
    grad2 = [(1.6376284958963165, 1.4190960600755023), (2.3050074818790436, -1.4238665362027818), (-2.1502120154461637, 2.068167990424798), (-1.6487191451588468, -1.4735402966701918),
             (-0.6121530315023754, -0.8471663497610038), (1.1041391731588124, 0.28777539893560494), (-1.254701781583775, 1.1437429101300747), (1.01614858834246, -1.4441140810004498)]
    ts = AdaptivePort.delaunay_triangles(p)
    @test Set(Tuple.(sort.(collect.(ts)))) == Set(keys(ref))
    losses = default_loss(p, v, ts)
    @test all(i -> isapprox(losses[i], ref[Tuple(sort(collect(ts[i])))]; rtol=1e-12), eachindex(ts))
    comp(c) = [z[c] for z in v]
    range_max = maximum(c -> maximum(comp(c)) - minimum(comp(c)), 1:3)
    g = AdaptivePort.estimate_gradients(p, comp(2) ./ range_max, ts)
    @test all(i -> all(isapprox.(g[i], grad2[i]; atol=1e-14)), eachindex(p))
    # the vector deviation is the sum of the component deviations at a common normalisation
    @test AdaptivePort.deviations(p, v, ts) ≈ sum(c -> AdaptivePort.component_deviations(p, comp(c) ./ range_max, ts), 1:3)
end
