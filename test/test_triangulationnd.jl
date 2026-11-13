module TriND
using AdaptivePort, Test
using Random: Xoshiro
const A = AdaptivePort

standard_simplex(N) = [ntuple(_ -> 0.0, N); [ntuple(i -> Float64(i == j), N) for j in 1:N]]
simplex_volume(N) = 1 / factorial(N)

# Port of upstream's _check_triangulation_is_valid, plus: no simplex is flat, volumes are positive.
function check_valid(t)
    S = eltype(t.simplices)
    v2s = [Set{S}() for _ in t.vertices]
    for s in t.simplices, v in s
        push!(v2s[v], s)
    end
    @test v2s == t.vertex_to_simplices
    @test all(<=(2), values(A.face_counts(t.simplices)))
    @test A.hull(t) == Set(v for (f, c) in A.face_counts(t.simplices) if c == 1 for v in f)
    @test all(s -> A.volume(t, s) > 0, t.simplices)
end

# Delaunay property: no vertex strictly inside a circumsphere.
function check_empty_spheres(t)
    for s in t.simplices
        c, r = A.circumsphere(t.vertices[collect(s)])
        for (i, p) in enumerate(t.vertices)
            i in s && continue
            @test sqrt(sum(abs2, c .- p)) >= r * (1 - 1e-8)
        end
    end
end

function add_checked!(t, p; kwargs...)
    old = copy(t.simplices)
    deleted, created = A.add_point!(t, p; kwargs...)
    @test deleted == setdiff(old, t.simplices)
    @test created == setdiff(t.simplices, old)
    return deleted, created
end

function random_triangulation(N, n, seed)
    rng = Xoshiro(seed)
    return A.TriangulationND([ntuple(_ -> rand(rng), N) for _ in 1:n])
end

function A_combinations(items, k)
    k == 0 && return [Int[]]
    isempty(items) && return Vector{Int}[]
    return vcat([[items[1]; c] for c in A_combinations(items[2:end], k - 1)], A_combinations(items[2:end], k))
end

function run_tests()
    @testset "dim $N" for N in 1:4
        n1 = N + 1
        @testset "standard simplex" begin
            t = A.TriangulationND(standard_simplex(N))
            @test t.simplices == Set([Tuple(1:n1)])
            check_valid(t)
            @test A.volume(t, Tuple(1:n1)) ≈ simplex_volume(N)
        end
        @testset "invalid input throws" begin
            pts = standard_simplex(N)
            @test_throws ArgumentError A.TriangulationND(pts[1:(end - 1)])  # too few
            dep = [pts[1:N]; [ntuple(i -> 0.37 * i, N)]]  # N + 1 points in one hyperplane (1D: two equal points)
            if N > 1
                dep[end] = ntuple(i -> sum(p[i] for p in pts[1:N]) / N, N)
                @test_throws ArgumentError A.TriangulationND(dep)
            end
            N == 1 && @test_throws ArgumentError A.TriangulationND([pts[1], pts[1]])  # coincident
            N > 1 && @test_throws ArgumentError A.TriangulationND([pts[1:N]; [pts[1] .+ 2 .* (pts[2] .- pts[1])]; [pts[N + 1]]][1:(N + 1)])
            t = A.TriangulationND(pts)
            @test_throws ArgumentError A.add_point!(t, pts[2])  # already a vertex
            @test length(t.vertices) == n1 && length(t.simplices) == 1  # unchanged by the failed insertion
        end
        @testset "outside the circumsphere in the positive orthant" begin
            t = A.TriangulationND(standard_simplex(N))
            add_checked!(t, ntuple(_ -> 1.1, N))
            s1, s2 = Tuple(1:n1), Tuple(2:(n1 + 1))
            check_valid(t)
            N > 1 && @test t.simplices == Set([s1, s2])
            @test A.hull(t) == Set(N == 1 ? [1, 3] : 1:(N + 2))
            @test t.vertex_to_simplices[1] == Set([s1])
            @test t.vertex_to_simplices[N + 2] == Set([s2])
        end
        @testset "outside in the negative orthant" begin
            t = A.TriangulationND(standard_simplex(N))
            add_checked!(t, ntuple(i -> Float64(i - N - 1), N))
            check_valid(t)
            @test length(t.simplices) == n1
            @test Tuple(1:n1) in t.simplices
            @test t.vertex_to_simplices[1] == t.simplices
            @test A.hull(t) == Set(2:(N + 2))
        end
        @testset "inside, simplex given: $given" for given in (false, true)
            t = A.TriangulationND(standard_simplex(N))
            add_checked!(t, ntuple(_ -> 0.1, N); simplex = given ? Tuple(1:n1) : nothing)
            check_valid(t)
            @test t.simplices == Set(Tuple([c; N + 2]) for c in A_combinations(collect(1:n1), N))
            @test sum(A.volumes(t)) ≈ simplex_volume(N)
        end
        N > 1 && @testset "on a face" begin
            pts = standard_simplex(N)
            t = A.TriangulationND(pts)
            add_checked!(t, ntuple(i -> sum(p[i] for p in pts[2:end]) / N, N))
            check_valid(t)
            @test t.simplices == Set(Tuple([1; c; N + 2]) for c in A_combinations(collect(2:n1), N - 1))
            @test sum(A.volumes(t)) ≈ simplex_volume(N)
        end
        @testset "on an edge" begin
            pts = standard_simplex(N)
            t = A.TriangulationND(pts)
            add_checked!(t, ntuple(i -> (pts[1][i] + pts[2][i]) / 2, N))
            check_valid(t)
            rest = collect(3:(N + 2))
            N > 1 && @test t.simplices == Set([Tuple([1; rest]), Tuple([2; rest])])
            @test sum(A.volumes(t)) ≈ simplex_volume(N)
        end
        @testset "collinear with the first edge" begin
            t = A.TriangulationND(standard_simplex(N))
            add_checked!(t, ntuple(i -> i == 1 ? 2.0 : 0.0, N))
            check_valid(t)
            N > 1 && @test t.simplices == Set([Tuple(1:n1), Tuple(2:(n1 + 1))])
        end
        N > 1 && @testset "coplanar with a face" begin
            pts = standard_simplex(N)
            t = A.TriangulationND(pts)
            add_checked!(t, ntuple(i -> 2 * sum(p[i] for p in pts[1:(end - 1)]), N))
            check_valid(t)
            @test t.simplices == Set([Tuple(1:n1), Tuple(2:(n1 + 1))])
        end
        @testset "inside the circumscribed sphere" begin
            t = A.TriangulationND(standard_simplex(N))
            add_checked!(t, ntuple(_ -> 0.6, N))
            check_valid(t)
            N > 1 && @test t.simplices == Set(Tuple([1; c; N + 2]) for c in A_combinations(collect(2:n1), N - 1))
        end
        @testset "random points: invariants after every insertion" begin
            rng = Xoshiro(N)
            t = A.TriangulationND(standard_simplex(N))
            for _ in 1:12
                add_checked!(t, ntuple(_ -> 1.4 * rand(rng) - 0.2, N))
                check_valid(t)
                check_empty_spheres(t)
                h = A.hull(t)
                @test length(A.faces(t)) >= length(t.simplices)
                @test all(v -> v in h || !isempty(t.vertex_to_simplices[v]), eachindex(t.vertices))
            end
        end
        @testset "volume is below the bounding box and the construction is deterministic" begin
            t1 = random_triangulation(N, 10, 1)
            @test sum(A.volumes(t1)) < 1 + 1e-8
            check_valid(t1)
            @test random_triangulation(N, 10, 1).simplices == t1.simplices
        end
        @testset "more than one simplex at construction" begin
            t = A.TriangulationND([standard_simplex(N); [ntuple(_ -> 1.1, N)]])
            check_valid(t)
            N > 1 && @test t.simplices == Set([Tuple(1:n1), Tuple(2:(n1 + 1))])
        end
        @testset "neighbours and containing" begin
            t = random_triangulation(N, 12, 5)
            for s in t.simplices
                nb = A.neighbors(t, s)
                @test all(x -> count(in(s), x) == N, nb)
                @test s in A.containing(t, s)
                @test length(nb) <= N + 1
            end
        end
        @testset "embedded volume" begin
            pts = standard_simplex(N)
            @test A.simplex_volume_in_embedding(pts) ≈ simplex_volume(N)
            emb = [(p..., 0.0, 0.0) for p in pts]  # same simplex in N + 2 dimensions
            @test A.simplex_volume_in_embedding(emb) ≈ simplex_volume(N)
        end
    end
    @testset "circumsphere of the standard 2-simplex" begin
        c, r = A.circumsphere([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
        @test all(c .≈ (0.5, 0.5)) && r ≈ sqrt(0.5)
    end
    @testset "embedded triangle in 3D" begin
        @test A.simplex_volume_in_embedding([(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 4.0, 0.0)]) ≈ 6
        @test A.simplex_volume_in_embedding([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]) == 0  # degenerate: clamped, not an error
    end
    @testset "transform: empty spheres in the transformed space" begin
        M = [1.0 0.0; 0.0 6.0]
        rng = Xoshiro(11)
        pts = [(rand(rng), rand(rng)) for _ in 1:30]
        t = A.TriangulationND(pts[1:3])
        for p in pts[4:end]
            A.add_point!(t, p; transform = M)
        end
        tv = [A.apply_transform(p, M) for p in t.vertices]
        for s in t.simplices
            c, r = A.circumsphere(tv[collect(s)])
            @test all(i -> i in s || sqrt(sum(abs2, c .- tv[i])) >= r * (1 - 1e-8), eachindex(tv))
        end
        @test sum(A.volumes(t)) > 0
    end
    @testset "N = 2 equals DelaunayTriangulation.jl" begin
        for seed in 1:5
            rng = Xoshiro(seed)
            pts = [(rand(rng), rand(rng)) for _ in 1:40]
            mine = A.TriangulationND(pts)
            theirs = Set(sort!(collect(t)) |> Tuple for t in A.delaunay_triangles(pts))
            @test Set(mine.simplices) == theirs
        end
    end
    @testset "N = 3 equals scipy.spatial.Delaunay" begin
        # Reference: `scripts` snippet run with `uv run --no-project --with scipy --with numpy python`:
        #   p = np.round(np.random.default_rng(0).random((12, 3)), 4); d = scipy.spatial.Delaunay(p)
        #   print the sorted 1-based vertex tuples of `d.simplices`.
        pts = [(0.637, 0.2698, 0.041), (0.0165, 0.8133, 0.9128), (0.6066, 0.7295, 0.5436), (0.9351, 0.8159, 0.0027), (0.8574, 0.0336, 0.7297), (0.1757, 0.8632, 0.5415), (0.2997, 0.4227, 0.0283), (0.1243, 0.6706, 0.6472), (0.6154, 0.3837, 0.9972), (0.9808, 0.6855, 0.6505), (0.6884, 0.3889, 0.1351), (0.7215, 0.5254, 0.3102)]
        scipy_simplices = [(1, 4, 5, 11), (1, 4, 7, 11), (1, 5, 7, 11), (2, 3, 6, 8), (2, 3, 6, 10), (2, 3, 8, 9), (2, 3, 9, 10), (2, 5, 8, 9), (2, 6, 7, 8), (3, 4, 6, 7), (3, 4, 6, 10), (3, 4, 7, 12), (3, 4, 10, 12), (3, 6, 7, 8), (3, 7, 8, 12), (3, 8, 9, 12), (3, 9, 10, 12), (4, 5, 10, 11), (4, 7, 11, 12), (4, 10, 11, 12), (5, 7, 8, 12), (5, 7, 11, 12), (5, 8, 9, 12), (5, 9, 10, 12), (5, 10, 11, 12)]
        t = A.TriangulationND(pts)
        @test sort!(collect(t.simplices)) == scipy_simplices
    end
end
end # module TriND

@testset "TriangulationND" begin
    TriND.run_tests()
end
