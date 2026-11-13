using Random, LinearAlgebra
import AdaptivePort: IntegratorLearner, DivergentIntegralError, Coeffs

# Test functions of upstream's `algorithm_4.py`. numpy gives NaN for `0 * sin(1 / 0)`, Julia would throw.
integrand_f0(x) = x == 0 ? NaN : x * sin(1 / x) * sqrt(abs(1 - x))
integrand_f7(x) = x^-0.5
integrand_f21(x) = sum(1 / cosh(20.0^i * (x - 2i / 10)) for i in 1:3)
integrand_f24(x) = floor(exp(x))
integrand_f63(x) = abs(x - 0.45)^0.987654321
integrand_fdiv(x) = abs(x - 0.987654321)^-1.1

# Run `n` asks of one point, as upstream's `run_integrator_learner`.
function run_integrator(f, a, b, tol, n)
    learner = IntegratorLearner(f; bounds = (a, b), tol)
    for _ in 1:n
        for x in ask(learner, 1)
            tell!(learner, x, f(x))
        end
    end
    return learner
end

@testset "IntegratorLearner" begin
    @testset "coefficients" begin
        for d in 1:4
            @test Coeffs.V_INV[d] * Coeffs.V[d] ≈ I atol = 1e-13
            @test Coeffs.XI[d] == -reverse(Coeffs.XI[d])
        end
                @test Coeffs.XI[1] ≈ [-1, -sqrt(0.5), 0, sqrt(0.5), 1] atol = 1e-15
        # the constant 1 has Legendre coefficient sqrt(2) on [-1, 1]
        @test (Coeffs.V_INV[2]*ones(9))[1] ≈ sqrt(2) atol = 1e-14
        @test Coeffs.ALPHA[1] ≈ sqrt(1 / 3) atol = 1e-16
    end

    # Reference: upstream's `algorithm_4.py` run with
    #   uv run --no-project --with adaptive --with scipy python ref.py
    # (`igral, err, n, ivals = algorithm_4(f, a, b, tol)`; ref.py loads tests/algorithm_4.py by path and prints the result).
    # The learner told `n` points one by one must give the same intervals. Tolerances: bounds equal, integrals 1e-14
    # (reference integrals are sums in a different order), errors rtol 1e-5 plus 1e-15 (the error of f63 is rounding noise).
    reference = Dict(
        "f0" => (f = integrand_f0, a = 0, b = 3, tol = 1e-5, igral = 1.98194117954329, err = 1.956354558992549e-05, n = 1129, nivals = 83),
        "f7" => (f = integrand_f7, a = 0, b = 1, tol = 1e-6, igral = 1.9999998579359646, err = 1.8561437335812635e-06, n = 693, nivals = 43),
        "f21" => (f = integrand_f21, a = 0, b = 1, tol = 1e-3, igral = 0.16310022131213361, err = 0.00011848806384952065, n = 191, nivals = 10),
        "f24" => (f = integrand_f24, a = 0, b = 3, tol = 1e-3, igral = 17.664696186312938, err = 0.01760261807495659, n = 4519, nivals = 99),
        "f63" => (f = integrand_f63, a = 0, b = 1, tol = 1e-10, igral = 0.25620504455630666, err = 2.5161495793738253e-11, n = 799, nivals = 17),
    )
    # (a, b, igral, err, depth) of the remaining intervals, sorted by a, for two cases
    ref_ivals_f21 = [
        (0.0, 0.25, 0.11999693137871906, 4.377329932503832e-06, 3),
        (0.25, 0.375, 0.032232748203006395, 5.651803125037423e-07, 2),
        (0.375, 0.390625, 0.0009272325768512659, 1.224947826740845e-05, 1),
        (0.390625, 0.3984375, 0.002659167447887413, 4.1813464867193257e-05, 1),
        (0.3984375, 0.40625, 0.00526045891736515, 3.7090401096442453e-06, 2),
        (0.40625, 0.421875, 0.0008407894537473736, 4.189052869304179e-05, 0),
        (0.421875, 0.4375, 0.0003181309831372597, 2.2160971780910145e-06, 0),
        (0.4375, 0.5, 0.0006172747481589977, 5.2312468596053296e-08, 1),
        (0.5, 0.75, 0.00024583120240551955, 1.1155393133399415e-05, 0),
        (0.75, 1.0, 1.6564008551502481e-06, 4.5923888713884084e-07, 0),
    ]
    ref_ivals_f63 = [
        (0.25, 0.375, 0.01760615296378747, 7.392132194556266e-17, 3),
        (0.375, 0.4375, 0.0028389498353191263, 1.275172195736566e-13, 3),
        (0.4375, 0.4453125, 7.116888866909791e-05, 9.917158402242335e-14, 2),
        (0.4453125, 0.44921875, 1.1475812188077392e-05, 2.0084574755414667e-12, 2),
        (0.44921875, 0.44970703125, 2.876841252504109e-07, 4.2947051538489874e-13, 1),
        (0.44970703125, 0.449951171875, 4.638837357055191e-08, 9.429519505459524e-13, 1),
        (0.449951171875, 0.449981689453125, 1.1628980131381787e-09, 5.6054966237025326e-14, 0),
        (0.449981689453125, 0.4499969482421875, 1.8751463790248426e-10, 7.05547135813273e-13, 0),
        (0.4499969482421875, 0.45001220703125, 9.22046729776972e-11, 7.070764772271278e-12, 0),
        (0.45001220703125, 0.4500732421875, 2.949320725149491e-09, 5.995182608574831e-14, 1),
        (0.4500732421875, 0.4501953125, 1.8290634517252326e-08, 8.816602045175157e-13, 0),
        (0.4501953125, 0.451171875, 7.296192584280613e-07, 1.2769546779486109e-13, 2),
        (0.451171875, 0.453125, 4.524838061398854e-06, 6.754924134299794e-12, 1),
        (0.453125, 0.46875, 0.00018049724408282354, 8.107165569333378e-15, 3),
        (0.46875, 0.5, 0.0011193794442327838, 1.5598203962134335e-12, 2),
        (0.5, 0.75, 0.0446524056830486, 2.005657016105993e-12, 3),
        (0.75, 1.0, 0.10735930346991844, 2.3236356090749907e-12, 2),
    ]
    ref_ivals = Dict("f21" => ref_ivals_f21, "f63" => ref_ivals_f63)

    @testset "same as reference implementation: $name" for (name, r) in reference
        learner = run_integrator(r.f, r.a, r.b, r.tol, r.n)
        res = result(learner)
        @test res.npoints == r.n
        @test res.integral ≈ r.igral atol = 1e-14 rtol = 0
        @test res.error ≈ r.err rtol = 1e-5 atol = 1e-15
        @test length(learner.ivals) == r.nivals
        @test AdaptivePort.done(learner)
        if haskey(ref_ivals, name)
            mine = sort(learner.ivals; by = iv -> iv.a)
            @test length(mine) == length(ref_ivals[name])
            for (iv, (a, b, igral, err, depth)) in zip(mine, ref_ivals[name])
                @test (iv.a, iv.b) == (a, b)
                @test iv.igral ≈ igral atol = 1e-14 rtol = 0
                @test iv.err ≈ err rtol = 1e-5 atol = 1e-15
                @test iv.depth == depth
            end
        end
    end

    @testset "divergent integral" begin
        # reference: algorithm_4 raises after 431 evaluations
        learner = IntegratorLearner(integrand_fdiv; bounds = (0, 1), tol = 1e-6)
        n = 0
        @test_throws DivergentIntegralError for _ in 1:2000
            for x in ask(learner, 1)
                n += 1
                tell!(learner, x, integrand_fdiv(x))
            end
        end
        @test n == 431
        @test occursin("divergent", sprint(showerror, DivergentIntegralError()))
    end

    @testset "choosing and adding points" begin
        learner = run_integrator(integrand_f24, 0, 3, 1e-10, 1000)
        @test npoints(learner) == 1000

        learner = IntegratorLearner(integrand_f24; bounds = (0, 3), tol = 1e-10)
        for x in ask(learner, 100)
            tell!(learner, x, integrand_f24(x))
        end
        @test npoints(learner) == 100
    end

    @testset "skip one point" begin
        f, tol = integrand_f24, 1e-10
        learner = IntegratorLearner(f; bounds = (0, 3), tol)
        xs = ask(learner, 17)
        skip_x = xs[2]
        for x in xs
            x == skip_x || tell!(learner, x, f(x))
        end
        for _ in 1:1000
            for x in ask(learner, 1)
                x == skip_x || tell!(learner, x, f(x))
            end
        end
        tell!(learner, skip_x, f(skip_x))
        # a learner with the same number of points gives the same integral
        learner2 = run_integrator(f, 0, 3, tol, 1017)
        @test npoints(learner) == npoints(learner2) == 1017
        @test result(learner).integral ≈ result(learner2).integral atol = 1e-9
    end

    @testset "tell in random order" begin
        for (f, a, b) in ((integrand_f21, 0, 1), (integrand_f24, 0, 3))
            learners = map((true, false)) do shuffled
                learner = IntegratorLearner(f; bounds = (a, b), tol = 1e-10)
                xs = ask(learner, 10000)
                shuffled && shuffle!(MersenneTwister(1), xs)
                for x in xs
                    tell!(learner, x, f(x))
                end
                learner
            end
            for learner in learners
                ivals = sort(AdaptivePort.approximating_intervals(learner); by = iv -> iv.a)
                @test ivals[1].a == a && ivals[end].b == b
                @test all(ivals[i].b == ivals[i+1].a for i in 1:length(ivals)-1)
                @test isfinite(result(learner).error)
            end
            r1, r2 = result.(learners)
            @test abs(r1.integral - r2.integral) <= r1.error + r2.error
        end
    end

    @testset "exact for a constant: stops on its own after the first interval" begin
        learner = IntegratorLearner(x -> 1.0; bounds = (-1, 1))
        # no interval is left to improve once the 33-point rule agrees with the 17-point one
        for _ in 1:33
            for x in ask(learner, 1)
                tell!(learner, x, 1.0)
            end
        end
        @test AdaptivePort.done(learner)
        @test result(learner).integral ≈ 2 atol = 1e-14
        @test_throws ErrorException ask(learner, 1)
    end

    @testset "removed interval: exp is done after 33 points" begin
        learner = IntegratorLearner(exp; bounds = (0, 1), tol = 1e-15)
        for _ in 1:33
            for x in ask(learner, 1)
                tell!(learner, x, exp(x))
            end
        end
        @test AdaptivePort.done(learner)
        @test result(learner).integral ≈ exp(1) - 1 atol = 1e-15
        @test_throws ErrorException ask(learner, 1)
    end

    @testset "non-finite values" begin
        # one infinity at the centre (upstream's f_one_with_nan): the interpolant of the other nodes is used
        learner = IntegratorLearner(x -> x == 0 ? Inf : 1.0; bounds = (-1, 1), tol = 1e-8)
        BlockingRunner(learner)
        @test result(learner).integral ≈ 2 atol = 1e-12
        @test isfinite(result(learner).error)
        # NaN at the endpoint of a singular integrand
        learner = run_integrator(integrand_f0, 0, 3, 1e-5, 200)
        @test isfinite(result(learner).integral) && isfinite(result(learner).error)
    end

    @testset "analytic integrals within the reported error" begin
        peak(x) = exp(-((x - 0.3) / 0.01)^2)
        cases = [
            (x -> 3.5, -1, 2, 10.5),
            (x -> x^5 - 2x^2 + 1, -1, 2, 7.5),
            (x -> sin(50x), 0, 3, (1 - cos(150)) / 50),
            (x -> cos(x)^2 * exp(-x), 0, 5, (1 - exp(-5)) / 2 + (1 + exp(-5) * (2sin(10) - cos(10))) / 10),
            (peak, 0, 1, 0.01 * sqrt(pi)),
        ]
        for (f, a, b, exact) in cases
            learner = IntegratorLearner(f; bounds = (a, b), tol = 1e-8)
            BlockingRunner(learner)
            res = result(learner)
            @test abs(res.integral - exact) <= max(res.error, 1e-14)
            @test res.error <= 1e-8 * abs(res.integral) + 1e-14
        end
    end

    @testset "interface" begin
        learner = IntegratorLearner(integrand_f21; bounds = (0, 1), tol = 1e-3)
        @test AdaptivePort.has_own_goal(learner)
        @test learner isa AbstractLearner
        @test npoints(learner) == 0
        @test result(learner) == (integral = 0.0, error = Inf, npoints = 0)
        @test loss(learner) == Inf
        @test !AdaptivePort.done(learner)

        # tell_pending = false leaves the learner as it was, also when more points than queued are asked for
        xs = ask(learner, 5; tell_pending = false)
        @test ask(learner, 5; tell_pending = false) == xs
        @test isempty(pending_points(learner))
        xs, improvements = AdaptivePort.ask_with_improvements(learner, 40; tell_pending = true)
        @test length(xs) == length(unique(xs)) == length(improvements) == 40
        @test Set(pending_points(learner)) == Set(xs[1:end])
        @test tell_pending!(learner, xs[1]) === learner
        @test remove_unfinished!(learner) === learner
        @test_throws ArgumentError tell!(learner, 0.123456789, 1.0)
        for x in xs
            tell!(learner, x, integrand_f21(x))
        end
        @test isempty(pending_points(learner))
        @test npoints(learner) == 40
        # tell! accepts told points again without harm
        before = result(learner)
        tell!(learner, xs[1], integrand_f21(xs[1]))
        @test result(learner) == before
        @test loss(learner; real = false) == loss(learner)
        @test AdaptivePort.done(learner; tol = Inf)
        @test !AdaptivePort.done(learner; tol = 0.0) || loss(learner) == 0
    end

    @testset "BlockingRunner stops without tol" begin
        learner = IntegratorLearner(integrand_f21; bounds = (0, 1), tol = 1e-3)
        BlockingRunner(learner)
        @test AdaptivePort.done(learner)
        @test result(learner).npoints == 191
        learner = IntegratorLearner(integrand_f21; bounds = (0, 1), tol = 1e-3)
        BlockingRunner(learner; parallel = true, batch_size = 4)
        @test AdaptivePort.done(learner)
        @test abs(result(learner).integral - 0.16310022131213361) <= result(learner).error
    end
end
