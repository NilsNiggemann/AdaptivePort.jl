using Random

# One noisy sample, fully determined by (seed, x).
noisy(mean_f, sigma_f) = (seed, x) -> mean_f(x) + sigma_f(x) * randn(Random.Xoshiro(hash((seed, x))))

# Manual ask/tell loop with one point per ask.
function run_steps!(l, f, n)
    for _ in 1:n
        p = only(ask(l, 1; tell_pending = false))
        tell!(l, p, f(p...))
    end
    return l
end

@testset "AverageLearner1D" begin
    @testset "Student-t quantile against scipy.stats.t.ppf" begin
        # scipy 1.x: [scipy.stats.t.ppf(1 - a, df) for a in (0.005, 0.1) for df in (1, 2, 5, 49)]
        reference = [63.656741162871526, 9.924843200918287, 4.032142983555228, 2.679951973631552, 3.0776835371752544, 1.8856180831641272, 1.4758840488244815, 1.29906878474775]
        computed = [AdaptivePort.StatsFuns.tdistinvcdf(df, 1 - a) for a in (0.005, 0.1) for df in (1, 2, 5, 49)]
        @test computed ≈ reference rtol = 1e-12
    end

    @testset "means, errors, loss and ask against upstream" begin
        # Reference: upstream AverageLearner1D(bounds=(-1, 1), min_samples=3, delta=0.2), told (seed, x, y) in this order
        # (see the data below), values printed by Python (repr, full precision). Tolerance 1e-12 relative.
        xs = [-1.0, -0.5, 0.0, 0.25, 1.0]
        told = [(0, -1.0, -0.9997861889268469), (1, -1.0, -0.9700346505117481), (2, -1.0, -1.027322989798817), (3, -1.0, -1.0889683881383225), (4, -1.0, -1.0453762827797675), (5, -1.0, -1.0990738597622414), (0, -0.5, -0.9565424968527111), (1, -0.5, -0.31650667537416355), (2, -0.5, -1.2327175574270952), (3, -0.5, -1.2968517480614006), (0, 0.0, 0.024492102509259914), (1, 0.0, 0.01784435040800304), (2, 0.0, 0.005270712449894928), (0, 0.25, 0.5691432265450516), (1, 0.25, 0.8395080932185308), (2, 0.25, 1.0568745982949992), (3, 0.25, 0.4450192757719883), (4, 0.25, 0.7109989116454474), (5, 0.25, 0.27791681801725965), (6, 0.25, 0.46142231802202005), (7, 0.25, 0.2957631286199932), (0, 1.0, 0.9528909780476589), (1, 1.0, 0.7464199079738545), (2, 1.0, 1.0541620760269355), (3, 1.0, 1.0312594215874402), (4, 1.0, 0.9625230153366042)]
        l = AverageLearner1D(nothing; min_samples = 3)
        for (seed, x, y) in told
            tell!(l, (seed, x), y)
        end
        r = result(l)
        @test r.x == xs
        @test r.nsamples == [6, 4, 3, 8, 5]
        @test r.y ≈ [-1.0384270599862904, -0.9506546194288426, 0.015869055122385962, 0.5820807962669112, 0.9494510797944987] rtol = 1e-12
        @test r.error ≈ [0.08255130724626589, 1.30779667588329, 0.055935921606991894, 0.33602430005821116, 0.25019636606629764] rtol = 1e-12
        @test loss(l) ≈ 0.5751432281523777 rtol = 1e-12
        @test npoints(l) == 5 && nsamples(l) == length(told)
        points, improvements = AdaptivePort.ask_with_improvements(l, 1; tell_pending = false)
        @test points == [(4, -0.5)]
        @test improvements ≈ [0.04378789908054814] rtol = 1e-12
        points, improvements = AdaptivePort.ask_with_improvements(l, 3; tell_pending = false)
        @test points == [(4, -0.5), (5, -0.5), (6, -0.5)]
        @test improvements ≈ fill(0.03374403813965143, 3) rtol = 1e-12
    end

    @testset "new x, pending points and loss(real = false) against upstream" begin
        l = AverageLearner1D(nothing; min_samples = 2)
        # Tolerance 1e-3: Learner1D scales the changed intervals with its last-recomputed y-scale, upstream with the current
        # one, which differs by 1.6e-4 here (the y-range grows by 2e-4 after the last full recompute).
        data = [(-1.0, [-1.0, -1.0001, -0.9999]), (0.0, [0.0, 0.0002, -0.0002]), (1.0, [1.0, 1.0001, 0.9998])]
        for (x, ys) in data, (s, y) in enumerate(ys)
            tell!(l, (s - 1, x), y)
        end
        @test result(l).error ≈ [0.0005730110893714369, 0.001146022178743, 0.0008752888970312385] rtol = 1e-10
        @test loss(l) ≈ 1.1177657491138473 rtol = 1e-3
        points, improvements = AdaptivePort.ask_with_improvements(l, 1; tell_pending = false)
        @test points == [(0, -0.5)]
        @test improvements ≈ [0.5588828745569236] rtol = 1e-3
        points, improvements = AdaptivePort.ask_with_improvements(l, 2; tell_pending = false)
        @test points == [(0, -0.5), (1, -0.5)]
        @test improvements ≈ fill(0.2794414372784618, 2) rtol = 1e-3
        @test only(ask(l, 1)) == (0, -0.5)
        @test loss(l; real = false) ≈ 0.5588828745569236 * 2 rtol = 1e-3  # one pending point halves the interval loss
        @test only(ask(l, 1)) == (0, 0.5)
        @test pending_points(l) == [(0, -0.5), (0, 0.5)]
        @test loss(l; real = false) ≈ 0.5588828745569236 rtol = 1e-3
        @test loss(l) ≈ 1.1177657491138473 rtol = 1e-3
        tell!(l, (0, -0.5), -0.99)
        @test pending_points(l) == [(0, 0.5)] && npoints(l) == 4
        remove_unfinished!(l)
        @test isempty(pending_points(l)) && isempty(l.inner.pending)
        @test ask(l, 1; tell_pending = false) == ask(l, 1; tell_pending = false)
    end

    @testset "undersampled points are resampled first" begin
        l = AverageLearner1D(nothing; min_samples = 3)
        tell!(l, (0, -1.0), 0.5)
        tell!(l, (1, -1.0), 0.4)
        tell!(l, (2, -1.0), 0.45)
        tell!(l, (0, 1.0), 0.2)
        tell!(l, (1, 1.0), 0.4)
        # upstream: ([(2, 1.0), (3, 1.0)], [inf, inf]) and loss 1.1180339887498947
        points, improvements = AdaptivePort.ask_with_improvements(l, 2; tell_pending = false)
        @test points == [(2, 1.0), (3, 1.0)] && all(isinf, improvements)
        @test loss(l) ≈ 1.1180339887498947 rtol = 1e-12
        @test result(l).error[2] ≈ 6.365674116287153 rtol = 1e-12
        # missing bounds are asked first, with one seed per requested point
        e = AverageLearner1D(nothing; bounds = (0, 2))
        @test ask(e, 3; tell_pending = false) == [(0, 0.0), (1, 0.0), (2, 0.0)]
        @test loss(e) == Inf && npoints(e) == 0 && nsamples(e) == 0 && ask(e, 0) == Tuple{Int,Float64}[]
    end

    @testset "input handling" begin
        @test_throws ArgumentError AverageLearner1D(nothing; delta = 0)
        @test_throws ArgumentError AverageLearner1D(nothing; alpha = 1.5)
        @test_throws ArgumentError AverageLearner1D(nothing; neighbor_sampling = 0)
        @test_throws ArgumentError AverageLearner1D(nothing; min_samples = 5, max_samples = 4)
        l = AverageLearner1D(nothing; min_samples = 2)
        @test_throws ArgumentError tell!(l, (0, 2.0), 1.0)
        tell!(l, (0, 0.0), 1.0)
        tell!(l, (0, 0.0), 100.0)  # repeated (seed, x) is ignored
        @test nsamples(l) == 1 && result(l).y == [1.0] && result(l).error == [Inf]
    end

    @testset "min_samples and max_samples" begin
        f = noisy(x -> tanh(5x), x -> 0.1)
        l = AverageLearner1D(f; min_samples = 4, max_samples = 12)
        run_steps!(l, f, 500)
        r = result(l)
        @test all(r.nsamples .>= 4) && all(r.nsamples .<= 12)
        @test npoints(l) > 10 && nsamples(l) == sum(r.nsamples) == 500
        @test issorted(r.x) && r.x[1] == -1 && r.x[end] == 1
    end

    @testset "constant mean, varying variance: more samples where the noise is large" begin
        f = noisy(x -> 0.0, x -> x < 0 ? 0.01 : 1.0)
        l = AverageLearner1D(f; min_samples = 5)
        run_steps!(l, f, 1500)
        r = result(l)
        quiet, loud = r.nsamples[r.x .< 0], r.nsamples[r.x .> 0]
        @test sum(loud) > 5 * sum(quiet)
        @test all(abs.(r.y[r.x .> 0]) .< 3 .* r.error[r.x .> 0])
    end

    @testset "smooth mean, constant noise" begin
        f = noisy(x -> sin(3x), x -> 0.1)
        l = AverageLearner1D(f; min_samples = 5)
        run_steps!(l, f, 1500)
        r = result(l)
        @test npoints(l) > 10
        @test maximum(abs.(r.y .- sin.(3 .* r.x)) ./ r.error) < 4  # the true mean lies within the confidence interval
        @test maximum(r.nsamples) < 0.5 * nsamples(l)  # no point hogs the budget
    end

    @testset "a localised peak attracts new x" begin
        f = noisy(x -> exp(-(x - 0.5)^2 / 0.003), x -> 0.01)
        l = AverageLearner1D(f; min_samples = 3)
        run_steps!(l, f, 1500)
        x = result(l).x
        near_peak = count(abs.(x .- 0.5) .< 0.15)
        @test near_peak > 0.3 * length(x)  # 15% of the domain
    end

    @testset "runners: max_npoints counts distinct x, tol works" begin
        f = noisy(x -> tanh(5x), x -> 0.05)
        l = AverageLearner1D(f; min_samples = 3)
        BlockingRunner(l; max_npoints = 12)
        @test npoints(l) == 12 && nsamples(l) >= 3 * 12 - 3
        @test isempty(pending_points(l))
        l = AverageLearner1D(f; min_samples = 3)
        BlockingRunner(l; tol = 0.1)
        @test loss(l) <= 0.1 && npoints(l) > 5
        @test_throws ArgumentError BlockingRunner(AverageLearner1D(f))
    end
end
