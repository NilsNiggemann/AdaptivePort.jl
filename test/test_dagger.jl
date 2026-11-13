using Dagger, Distributed

@testset "DaggerExecutor" begin
    @testset "constructor and default_ntasks" begin
        @test default_ntasks(DaggerExecutor()) >= 1
        @test DaggerExecutor(; scope = Dagger.ProcessScope(1)) isa DaggerExecutor
        @test default_ntasks(DaggerExecutor(; scope = Dagger.ProcessScope(1))) >= 1
    end

    @testset "runner on the local process" begin
        l = Learner1D(x -> tanh(50x); bounds = (-1, 1))
        r = BlockingRunner(l; max_npoints = 40, executor = DaggerExecutor())
        @test status(r) === :finished
        @test npoints(l) >= 40
        @test fetch(DaggerExecutor()(x -> 2x, (3.0,))) == 6.0
    end

    @testset "throwing f gives EvaluationError with the original exception" begin
        l = Learner1D(x -> x > 0 ? throw(DomainError(x, "boom")) : x; bounds = (-1, 1))
        err = try
            BlockingRunner(l; max_npoints = 10, executor = DaggerExecutor())
            nothing
        catch e
            e
        end
        @test err isa EvaluationError
        @test err.exception.ex isa DomainError
        @test occursin("test_dagger", string(err.exception.processed_bt))
        @test err.point[1] > 0
    end

    @testset "with addprocs(2)" begin
        ws = addprocs(2)
        try
            @everywhere ws using Dagger
            Dagger.addprocs!(Dagger.Sch.eager_context(), ws)
            # named functions defined everywhere: anonymous ones are sent by value and intermittently failed to
            # deserialize on the two-thread workers (MethodError on the deserialized closure type)
            @everywhere begin
                worker_id(x) = Distributed.myid()
                square(x) = x^2
                remote_fail(x) = error("remote")
            end
            ex = DaggerExecutor(; scope = Dagger.UnionScope([Dagger.ProcessScope(w) for w in ws]))
            @test default_ntasks(ex) >= 2
            pids = Set(fetch(ex(worker_id, (Float64(i),))) for i in 1:20)
            @test pids == Set(ws)
            l = Learner1D(square; bounds = (-1, 1))
            r = BlockingRunner(l; max_npoints = 20, executor = ex)
            @test status(r) === :finished
            @test_throws EvaluationError BlockingRunner(Learner1D(remote_fail; bounds = (-1, 1)); max_npoints = 5, executor = ex)
        finally
            rmprocs(ws)
        end
    end
end
