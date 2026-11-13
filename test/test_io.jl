@testset "data export and saving" begin
    # tell a learner `n` points of f by hand
    function fill!(l, f, n)
        for x in ask(l, n)
            tell!(l, x, f(x...))
        end
        return l
    end
    # the learner keeps sampling after a reload
    function keeps_sampling(l)
        n = length(first(to_table(l)))
        pts = ask(l, 3)
        m = count(k -> startswith(string(k), r"(y|value)_"), keys(to_table(l)))
        foreach(p -> tell!(l, p, m == 0 ? 1.0 : ntuple(_ -> 1.0, m)), pts)
        return !isempty(pts) && length(first(to_table(l))) > n
    end
    leftovers(dir) = filter(!in(["data.csv", "data.jls"]), readdir(dir))

    f1(x) = tanh(50x)
    f2(x, y) = x^2 + sin(3y)
    f3(x, y, z) = x + y * z

    makers = [
        ("Learner1D", () -> Learner1D(f1; bounds = (-1, 1)), l -> fill!(l, f1, 25), (:x, :y)),
        ("Learner1D vdim=2", () -> Learner1D(x -> (x, x^2); bounds = (-1, 1), vdim = 2), l -> fill!(l, x -> (x, x^2), 20), (:x, :y_1, :y_2)),
        ("Learner2D", () -> Learner2D(f2; bounds = ((-1, 1), (-1, 1))), l -> fill!(l, f2, 40), (:x, :y, :value)),
        ("Learner2D vdim=2", () -> Learner2D((x, y) -> (x, y); bounds = ((-1, 1), (-1, 1)), vdim = 2), l -> fill!(l, (x, y) -> (x, y), 30), (:x, :y, :value_1, :value_2)),
        ("LearnerND N=3", () -> LearnerND(f3; bounds = ((-1, 1), (-1, 1), (-1, 1))), l -> fill!(l, f3, 30), (:x, :y, :z, :value)),
        ("LearnerND N=2 vdim=2", () -> LearnerND((x, y) -> (x, y); bounds = ((-1, 1), (-1, 1)), vdim = 2), l -> fill!(l, (x, y) -> (x, y), 30), (:x, :y, :value_1, :value_2)),
        ("LearnerND N=4", () -> LearnerND((a, b, c, d) -> a + b * c * d; bounds = ntuple(_ -> (-1, 1), 4)), l -> fill!(l, (a, b, c, d) -> a + b * c * d, 30), (:x_1, :x_2, :x_3, :x_4, :value)),
        ("AverageLearner", () -> AverageLearner(nothing; atol = 0.1), l -> foreach(s -> tell!(l, s, sin(s)), [3, 0, 7, 1, 2]), (:seed, :y)),
        ("AverageLearner1D", () -> AverageLearner1D(nothing; min_samples = 3), l -> foreach(p -> tell!(l, p, p[2]^2 + sin(p[1])), [(s, x) for x in (-1, 1, 0) for s in 0:3]), (:seed, :x, :y)),
        ("IntegratorLearner", () -> IntegratorLearner(x -> exp(x); bounds = (0, 1), tol = 1e-8), l -> foreach(_ -> fill!(l, exp, 1), 1:12), (:x, :y)),
        ("SequenceLearner", () -> SequenceLearner(x -> x^2; sequence = [10.0, 20.0, 30.0, 40.0]), l -> foreach(i -> tell!(l, (i, 10.0 * i), 100.0 * i^2), [3, 1, 4]), (:index, :x, :y)),
    ]

    @testset "$name" for (name, make, fill_learner, cols) in makers
        l = make()
        fill_learner(l)
        @test npoints(l) > 0
        t = to_table(l)
        @test keys(t) == cols
        @test allequal(length.(values(t)))
        @test length(first(t)) == (l isa AverageLearner1D ? 12 : npoints(l))
        @test to_table(l) == t  # no mutation, no change
        @test to_table(make()) isa NamedTuple && all(isempty, to_table(make()))
        mktempdir() do dir
            for file in ("data.csv", "data.jls")
                path = joinpath(dir, file)
                @test save_data(path, l) == path
                @test isfile(path)
                l2 = make()
                @test load_data!(l2, path) === l2
                @test npoints(l2) == npoints(l)
                if l isa AverageLearner  # running sums depend on the summation order in the last bits
                    @test all(values(result(l2)) .≈ values(result(l)))
                elseif l isa Union{Learner1D, SequenceLearner, IntegratorLearner}
                    @test result(l2) == result(l)
                else
                    @test to_table(l2) == t  # same rows, possibly in the same order
                end
                @test keeps_sampling(l2)
            end
            @test isempty(leftovers(dir))
        end
    end

    @testset "values and order" begin
        l = Learner1D(x -> 2x; bounds = (0, 1))
        foreach(x -> tell!(l, x, 2x), (1.0, 0.0, 0.25))
        @test to_table(l) == (x = [0.0, 0.25, 1.0], y = [0.0, 0.5, 2.0])
        l2 = Learner2D((x, y) -> x + y; bounds = ((0, 1), (0, 1)))
        tell!(l2, (1.0, 0.0), 1.0)
        tell!(l2, (0.0, 1.0), 5.0)
        @test to_table(l2) == (x = [1.0, 0.0], y = [0.0, 1.0], value = [1.0, 5.0])
        s = SequenceLearner(identity; sequence = ["a", "b"])
        tell!(s, (2, "b"), 7)
        @test to_table(s) == (index = [2], x = ["b"], y = [7])
        a = AverageLearner(nothing; atol = 1)
        tell!(a, 5, 1.5)
        tell!(a, 2, 0.5)
        @test to_table(a) == (seed = [2, 5], y = [0.5, 1.5])
    end

    @testset "lossless CSV with NaN and Inf" begin
        l = Learner2D((x, y) -> 0.0; bounds = ((0, 1), (0, 1)))
        vals = [0.1 + 0.2, NaN, Inf, -Inf, 1e-300]
        for (i, v) in enumerate(vals)
            tell!(l, (i / 7, 1 / 3), v)
        end
        mktempdir() do dir
            path = joinpath(dir, "data.csv")
            save_data(path, l)
            lines = readlines(path)
            @test lines[1] == "x,y,value"
            @test lines[2] == "0.14285714285714285,0.3333333333333333,0.30000000000000004"
            l2 = Learner2D((x, y) -> 0.0; bounds = ((0, 1), (0, 1)))
            load_data!(l2, path)
            @test isequal(to_table(l2), to_table(l))
        end
    end

    @testset "overwrite is atomic" begin
        mktempdir() do dir
            path = joinpath(dir, "data.jls")
            l = fill!(Learner1D(f1; bounds = (-1, 1)), f1, 5)
            save_data(path, l)
            fill!(l, f1, 5)
            save_data(path, l)
            @test readdir(dir) == ["data.jls"]
            @test npoints(load_data!(Learner1D(f1; bounds = (-1, 1)), path)) == npoints(l)
        end
    end

    @testset "errors" begin
        l = fill!(Learner1D(f1; bounds = (-1, 1)), f1, 5)
        mktempdir() do dir
            missing_path = joinpath(dir, "nope.jls")
            e = try load_data!(l, missing_path) catch e e end
            @test e isa ArgumentError && occursin(missing_path, e.msg)
            # wrong header
            path = joinpath(dir, "data.csv")
            save_data(path, fill!(Learner2D(f2; bounds = ((-1, 1), (-1, 1))), f2, 10))
            e = try load_data!(Learner1D(f1; bounds = (-1, 1)), path) catch e e end
            @test e isa ArgumentError && occursin("expected", e.msg) && occursin("found", e.msg) && occursin(":value", e.msg)
            path = joinpath(dir, "data.jls")
            save_data(path, fill!(Learner1D(x -> (x, x); bounds = (-1, 1), vdim = 2), x -> (x, x), 5))
            @test_throws ArgumentError load_data!(Learner1D(f1; bounds = (-1, 1)), path)
            # foreign .jls
            write(joinpath(dir, "other.jls"), "text")
            @test_throws Exception load_data!(l, joinpath(dir, "other.jls"))
            # non-numeric to CSV
            s = SequenceLearner(identity; sequence = ["a", "b"])
            tell!(s, (1, "a"), 1)
            e = try save_data(joinpath(dir, "s.csv"), s) catch e e end
            @test e isa ArgumentError && occursin("numbers only", e.msg) && occursin(".jls", e.msg)
            @test !ispath(joinpath(dir, "s.csv"))
            # but .jls works
            save_data(joinpath(dir, "s.jls"), s)
            s2 = SequenceLearner(identity; sequence = ["a", "b"])
            load_data!(s2, joinpath(dir, "s.jls"))
            @test result(s2) == result(s)
            @test isempty(filter(!in(["data.csv", "data.jls", "other.jls", "s.jls"]), readdir(dir)))
        end
    end
end
