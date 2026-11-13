@testset "SequenceLearner" begin
    @testset "order, told out of order, pending then removed" begin
        l = SequenceLearner(x -> x^2; sequence = [10, 20, 30, 40, 50])
        @test ask(l, 2) == [(1, 10), (2, 20)]
        @test pending_points(l) == [(1, 10), (2, 20)]
        @test loss(l) == 1.0 && loss(l; real = false) == 0.6
        tell!(l, (4, 40), 1600)  # out of order, never asked
        @test ask(l, 5) == [(3, 30), (5, 50)]
        @test loss(l; real = false) == 0.0 && loss(l) == 0.8 && !done(l)
        remove_unfinished!(l)
        @test isempty(pending_points(l))
        @test ask(l, 5; tell_pending = false) == [(1, 10), (2, 20), (3, 30), (5, 50)]
        @test AdaptivePort.ask_with_improvements(l, 2)[2] == [0.2, 0.2]
        tell!(l, (2, 20), 400)
        @test result(l) == (indices = [2, 4], values = [400, 1600])  # partial result
        tell!(l, (1, 10), 100)
        tell!(l, (3, 30), 900)
        tell!(l, (5, 50), 2500)
        @test done(l) && loss(l) == 0 && npoints(l) == 5
        @test result(l) == (indices = 1:5, values = [100, 400, 900, 1600, 2500])
        @test isempty(ask(l, 3))
    end

    @testset "empty sequence" begin
        l = SequenceLearner(identity; sequence = Int[])
        @test done(l) && loss(l) == 0 && isempty(ask(l, 3)) && npoints(l) == 0
        @test result(l) == (indices = Int[], values = Any[])
        BlockingRunner(l)
    end

    @testset "duplicate elements and unhashable elements" begin
        l = SequenceLearner(x -> sum(values(x)); sequence = [Dict(1 => 1), Dict(1 => 1), Dict(1 => 1)])
        BlockingRunner(l)
        @test result(l).values == [1, 1, 1] && npoints(l) == 3
    end

    @testset "value_type" begin
        l = SequenceLearner(x -> x / 2; sequence = 1:4, value_type = Float64)
        BlockingRunner(l)
        @test result(l).values isa Vector{Float64} && result(l).values == [0.5, 1, 1.5, 2]
    end

    @testset "BlockingRunner stops on its own, the wrapper ignores the index" begin
        seen = []
        l = SequenceLearner(x -> (push!(seen, x); x + 1); sequence = [3, 1, 2])
        @test l.f(99, 5) == 6
        empty!(seen)
        r = BlockingRunner(l; batch_size = 2)
        @test status(r) === :finished && seen == [3, 1, 2]
        @test result(l).values == [4, 2, 3]
    end

    @testset "done with tol and tell_pending! on told" begin
        l = SequenceLearner(identity; sequence = 1:4)
        tell!(l, (1, 1), 1)
        tell_pending!(l, (1, 1))  # already told: stays unpending
        @test isempty(pending_points(l))
        @test done(l; tol = 0.75) && !done(l; tol = 0.5)
    end
end
