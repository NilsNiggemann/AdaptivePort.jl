@testset "examples" begin
    # Fast examples that need only AdaptivePort, each in a fresh module so their globals do not leak.
    # Not run here: learner2d, learnernd, vector (long); plotting (CairoMakie, `--project=examples`);
    # distributed, dagger, mpi (extra packages and processes).
    examples = ["learner1d", "integrator", "balancing", "datasaver", "average_learner", "average_learner1d",
        "sequence", "runners", "save_load"]
    for name in examples
        @testset "$name" begin
            path = joinpath(@__DIR__, "..", "examples", name * ".jl")
            @test redirect_stdout(devnull) do
                Base.include(Module(), path)
                true
            end
        end
    end
end

@testset "read-only accessors leave the learner unchanged" begin
    l1 = Learner1D(x -> tanh(20x); bounds = (-1, 1))
    l2 = Learner2D((x, y) -> sin(3x) * y; bounds = ((-1, 1), (-1, 1)))
    for (l, f) in ((l1, l1.f), (l2, l2.f))
        for _ in 1:30
            for p in ask(l, 3)
                tell!(l, p, f(p...))
            end
        end
        ask(l, 2)                        # leave some pending points
        before = (result(l), to_table(l), pending_points(l), loss(l), npoints(l))
        result(l); to_table(l); l isa Learner2D && triangles(l)
        @test (result(l), to_table(l), pending_points(l), loss(l), npoints(l)) == before
        r = result(l)                    # results are copies
        r isa NamedTuple && foreach(v -> v isa Vector && empty!(v), values(r))
        @test npoints(l) == before[5] && result(l) == before[1]
        tris = l isa Learner2D ? triangles(l) : Tuple{Int,Int,Int}[]
        empty!(tris)
        @test l isa Learner2D ? !isempty(triangles(l)) : true
    end
end
