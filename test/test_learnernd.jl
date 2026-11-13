module LearnerNDTests
using AdaptivePort, Test
const A = AdaptivePort

# Reference from upstream (python-adaptive, `LearnerND`), produced with
#   uv run --no-project --with adaptive --with scipy --with numpy python ref_nd.py
# which tells the sorted box corners plus the extra points below, then records 12 single `ask(1)` points (each told
# right after), a batch `ask(6)` (pending, then told), another `ask(6)`, and `ask(4, tell_pending=False)` twice.
# Entries are (point, loss improvement); upstream rounds losses to 8 digits.
REF = Dict(
  "2d_corners" => (seq = [((-0.3333333333333333, 0.3333333333333333), 0.6127695928848328), ((-0.1111111111111111, -0.5555555555555556), 0.5230385918637511), ((0.4444444444444444, 0.2222222222222222), 0.44373417051624026), ((0.33333333333333337, 0.6666666666666666), 0.31149993778862556), ((1.0, 0.0), 0.24890319731527583), ((0.0, -1.850371707708594e-17), 0.2356781894518594), ((-1.0, 0.0), 0.2345246969472267), ((-0.7037037037037037, -0.5185185185185185), 0.16813177570447635), ((0.7222222222222222, 0.6111111111111112), 0.147543925303732), ((0.6296296296296297, -0.5185185185185185), 0.14066452278522595), ((0.0, -1.0), 0.1282692552457703), ((-0.5679012345679012, -0.8395061728395062), 0.13196954793553778)],
    batch = [((-0.345679012345679, -0.06172839506172839), 0.12132412509434011), ((-0.3333333333333333, 0.8333333333333333), 0.1202648755472261), ((0.16666666666666669, 0.3333333333333333), 0.12003671863755024), ((0.35802469135802467, -0.09876543209876543), 0.10548010953618704), ((-0.35185185185185186, -0.25925925925925924), 0.10540036161849686), ((-0.7777777777777777, 0.4444444444444444), 0.09463497933979165)],
    batch2 = [((-0.46090534979423864, -0.6378600823045267), 0.08899380892905676), ((-0.757201646090535, -0.786008230452675), 0.07946073621949445), ((0.0823045267489712, -0.21810699588477367), 0.07886440791289169), ((0.5, 0.5), 0.07348107235631152), ((0.6666666666666667, 0.8333333333333333), 0.07249207019542123), ((0.5, -1.0), 0.07171043743019807)],
    nopending = [(-0.15432098765432098, -0.271604938271605), (-1.0, -0.5), (-0.40740740740740744, -0.537037037037037), (-0.16666666666666666, 0.49999999999999994)],
    nopending2 = [(-0.15432098765432098, -0.271604938271605), (-1.0, -0.5), (-0.40740740740740744, -0.537037037037037), (-0.16666666666666666, 0.49999999999999994)],
    loss = 0.08899380892905676),
  "2d_extra" => (seq = [((-0.09333333333333334, -0.36000000000000004), 0.3686255825659012), ((0.81, 0.31), 0.2545367169723387), ((-0.725, -0.15000000000000002), 0.2510140933268308), ((-1.0, 0.0), 0.21563557139442527), ((-0.6061111111111112, -0.5033333333333334), 0.19416846855131417), ((1.0, 0.0), 0.18403047752254112), ((0.0, -1.0), 0.1676578207271372), ((0.55, 0.65), 0.16098113141493825), ((-0.23944444444444443, -0.07000000000000002), 0.14344756377461454), ((-0.5353703703703704, -0.8344444444444444), 0.12424467165090462), ((0.51, 0.07666666666666666), 0.12377495860544371), ((0.17222222222222225, 0.005555555555555536), 0.1187181200090636)],
    batch = [((0.0, 1.0), 0.1025138410400822), ((0.3866666666666667, 0.3422222222222222), 0.0958750136556006), ((0.54, -0.7933333333333333), 0.09370306985194726), ((-0.3143518518518519, -0.5972222222222222), 0.09134209403141857), ((-0.312962962962963, -0.31111111111111117), 0.08179281244416947), ((-0.8030555555555556, -0.7516666666666667), 0.07993574839421357)],
    batch2 = [((0.905, 0.655), 0.07676884983864005), ((-0.8166666666666668, 0.5666666666666667), 0.07020937292716851), ((0.010925925925925941, 0.0785185185185185), 0.06936097738663453), ((-0.053518518518518514, -0.1414814814814815), 0.06831798296654336), ((1.0, -0.5), 0.0658619626321424), ((-0.19648148148148148, 0.31), 0.06129676982560881)],
    nopending = [(-0.47148148148148145, 0.15999999999999998), (0.35555555555555557, -0.5111111111111111), (-0.725, 0.18333333333333332), (0.5, 1.0)],
    nopending2 = [(-0.47148148148148145, 0.15999999999999998), (0.35555555555555557, -0.5111111111111111), (-0.725, 0.18333333333333332), (0.5, 1.0)],
    loss = 0.07676884983864005),
  "3d_extra" => (seq = [((0.0, 0.0, 1.0), 0.08726210407284236), ((0.0, 0.0, -1.0), 0.08172628280911658), ((-0.19, -0.69, -0.245), 0.07785671348707207), ((0.81, 0.31, -0.245), 0.06827056034923508), ((-0.725, -0.15000000000000002, -0.33499999999999996), 0.06820763003008867), ((1.0, 1.0, 0.0), 0.05112972595870639), ((-1.0, 0.0, 1.0), 0.05102172394364717), ((0.0, -1.0, 1.0), 0.04945453182380801), ((0.56, -0.43999999999999995, -0.245), 0.0423380085067232), ((0.0, 1.0, -1.0), 0.04005158688316344), ((-1.0, 1.0, 0.0), 0.03961279506707654), ((1.0, 0.0, -1.0), 0.03831095570008834)],
    batch = [((-0.5, -1.0, 0.0), 0.03550680793380829), ((0.16250000000000003, 0.5, 0.5325), 0.0339453066518136), ((0.0, 1.0, 1.0), 0.03294361990814329), ((0.0, -1.0, -1.0), 0.03289612398334866), ((-0.595, -0.345, 0.3775), 0.03205536948310051), ((-1.0, -1.0, 0.0), 0.03164948207762043)],
    batch2 = [((1.0, -1.0, 0.0), 0.038493035768662795), ((0.275, 0.85, 0.165), 0.029875623980585395), ((-1.0, 0.0, -1.0), 0.029243168807085246), ((-0.65625, 0.5375, -0.38375), 0.026618041812401367), ((1.0, 0.0, 1.0), 0.025152309667989387), ((0.81, 0.31, 0.755), 0.024718528841450527)],
    nopending = [(-0.016249999999999987, -0.27875, 0.110625), (-0.041249999999999995, -0.60375, 0.410625), (0.07187500000000002, 0.018750000000000003, 0.305), (-0.475, 0.825, -0.55)],
    nopending2 = [(-0.016249999999999987, -0.27875, 0.110625), (-0.041249999999999995, -0.60375, 0.410625), (0.07187500000000002, 0.018750000000000003, 0.305), (-0.475, 0.825, -0.55)],
    loss = 0.038493035768662795),
  "2d_vec" => (seq = [((-0.09333333333333334, -0.36000000000000004), 0.49651371276398776), ((-0.725, -0.15000000000000002), 0.34319169444656344), ((-0.23944444444444443, -0.07000000000000002), 0.3455467232949568), ((0.0, 1.0), 0.30465389593613124), ((0.81, 0.31), 0.28285628864762236), ((-1.0, 0.0), 0.2785561993987593), ((0.20888888888888887, -0.1466666666666667), 0.27563296134948834), ((1.0, 0.0), 0.25778933074066396), ((0.02314814814814815, 0.027777777777777762), 0.2437724271100944), ((-0.6061111111111112, -0.5033333333333334), 0.22121720650740243), ((0.0, -1.0), 0.20384769967501037), ((0.54, -0.7933333333333333), 0.16251195880088828)],
    batch = [((0.6033333333333334, 0.77), 0.1590320019475613), ((0.405, 0.655), 0.1501830537716556), ((0.37296296296296294, 0.1544444444444444), 0.1473757350679157), ((-0.31296296296296294, -0.31111111111111117), 0.12386552154134393), ((-1.0, -0.5), 0.12252169279312727), ((-0.5, 1.0), 0.1219176825011514)],
    batch2 = [((-0.5353703703703704, -0.8344444444444444), 0.10484714734374728), ((-0.8166666666666668, 0.5666666666666667), 0.10377824536409168), ((0.2926543209876543, 0.3698148148148148), 0.10110085422304704), ((-0.2331481481481482, -0.6211111111111112), 0.09580678892256624), ((0.5, -1.0), 0.09157666973489852), ((-0.19648148148148148, 0.31), 0.08810390138867392)],
    nopending = [(0.905, 0.655), (1.0, -0.5), (-0.11666666666666665, 0.6666666666666666), (0.1488888888888889, -0.7177777777777777)],
    nopending2 = [(0.905, 0.655), (1.0, -0.5), (-0.11666666666666665, 0.6666666666666666), (0.1488888888888889, -0.7177777777777777)],
    loss = 0.10484714734374728),
)

f2(x, y) = sin(3x) * cos(2y) + tanh(5 * (x - y))
f3(x, y, z) = x * y + tanh(4z) + sin(x + y)
fv(x, y) = (sin(3x) + y, tanh(5 * (x - y)))
const EXTRA2 = [(0.1, 0.3), (-0.45, 0.7), (0.62, -0.38)]
const EXTRA3 = [(0.1, 0.3, -0.2), (-0.45, 0.7, 0.33), (0.62, -0.38, 0.51)]

# 2D learner on the default box [-1, 1]^2.
L2(f; kwargs...) = LearnerND(f; bounds = ((-1, 1), (-1, 1)), kwargs...)

# Tell all points of `ps` with `f`.
tell_all!(l, f, ps) = foreach(p -> tell!(l, p, f(p...)), ps)

# Ask in batches of `batch` and tell until `n` points are evaluated.
function run!(l, f, n; batch = 4)
    while npoints(l) < n
        for p in ask(l, batch)
            tell!(l, p, f(p...))
        end
    end
    return l
end

# Structural invariants of the learner and its triangulation.
function check_invariants(l)
    pts = l.points
    @test length(l.values) == length(pts) && all(i -> l.index[pts[i]] == i, eachindex(pts))
    @test length(Set(pts)) == length(pts)
    @test isdisjoint(l.pending, Set(pts))
    all(p -> all(i -> l.bounds[i][1] <= p[i] <= l.bounds[i][2], 1:length(p)), pts) || @test false
    tri = l.tri
    tri === nothing && return
    S = eltype(tri.simplices)
    @test tri.vertices == pts
    @test Set(keys(l.losses)) == tri.simplices
    v2s = [Set{S}() for _ in tri.vertices]
    for s in tri.simplices, v in s
        push!(v2s[v], s)
    end
    @test v2s == tri.vertex_to_simplices
    @test all(<=(2), values(A.face_counts(tri.simplices)))
    if A.bounds_done(l)
        box = prod(b -> b[2] - b[1], l.bounds)
        @test sum(A.volumes(tri)) ≈ box rtol = 1e-8
    end
    @test all(isfinite, values(l.losses)) && all(>=(0), values(l.losses))
end

function run_tests()
    @testset "construction" begin
        l = LearnerND(f3)
        @test npoints(l) == 0 && loss(l) == Inf && isempty(simplices(l)) && isempty(pending_points(l))
        @test l.bounds == ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
        l2 = LearnerND(f2; bounds = ((0, 1), (-2, 5)))
        @test l2.bounds == ((0.0, 1.0), (-2.0, 5.0))
        @test_throws ArgumentError LearnerND(f2; bounds = ((0, 1), (2, 2)))
        @test_throws ArgumentError LearnerND(f2; bounds = ((0, Inf), (0, 1)))
        @test_throws ArgumentError LearnerND(f2; vdim = 0)
        @test result(l2) == (points = NTuple{2, Float64}[], values = Float64[])
    end

    @testset "corners first, in upstream order" begin
        l = LearnerND(f2; bounds = ((0, 1), (0, 2)))
        @test ask(l, 4) == [(0.0, 0.0), (0.0, 2.0), (1.0, 0.0), (1.0, 2.0)]
        @test pending_points(l) == [(0.0, 0.0), (0.0, 2.0), (1.0, 0.0), (1.0, 2.0)]
        l3 = LearnerND(f3)
        @test ask(l3, 8) == sort!(vec(collect(Iterators.product((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)))))
        # corners are pending but nothing is told: a random point inside the box, reproducible
        p = only(ask(l3, 1))
        @test all(x -> -1 <= x <= 1, p)
        @test p == only(ask(LearnerND(f3; bounds = l3.bounds) |> l -> (ask(l, 8); l), 1))
    end

    @testset "telling" begin
        l = LearnerND(f2; bounds = ((0, 1), (0, 1)))
        @test tell!(l, (0.0, 0.0), 1) === l
        tell!(l, [1, 0], 2.0)  # any iterable point, integers allowed
        @test npoints(l) == 2 && result(l).points == [(0.0, 0.0), (1.0, 0.0)] && result(l).values == [1.0, 2.0]
        tell!(l, (0.0, 0.0), 99.0)  # repeated points are ignored, as upstream
        @test result(l).values[1] == 1.0
        @test_throws ArgumentError tell!(l, (1.5, 0.5), 0.0)
        @test_throws ArgumentError tell!(l, (0.5, -0.1), 0.0)
        @test_throws ArgumentError tell!(l, (0.5,), 0.0)
        @test_throws ArgumentError tell!(l, (0.5, 0.5), NaN)
        @test_throws ArgumentError tell!(l, (0.5, 0.5), [1.0, 2.0])
        @test_throws ArgumentError tell_pending!(l, (2.0, 2.0))
        @test npoints(l) == 2
        r = result(l)
        push!(r.points, (0.3, 0.3))
        @test npoints(l) == 2  # result returns copies
        # told points leave the pending set; pending points that are already told are ignored
        tell_pending!(l, (0.5, 0.5))
        @test pending_points(l) == [(0.5, 0.5)]
        tell!(l, (0.5, 0.5), 3.0)
        @test isempty(pending_points(l))
        tell_pending!(l, (0.5, 0.5))
        @test isempty(pending_points(l))
    end

    @testset "constant and linear functions" begin
        for (f, name) in ((x -> 3.0, "constant"), (x -> 2x[1] - x[2] + 0.5x[3], "linear"))
            g(x, y, z) = f((x, y, z))
            l = LearnerND(g)
            run!(l, g, 60)
            check_invariants(l)
            @test isfinite(loss(l))
            @test npoints(l) == 60
            if name == "linear"  # the linear function is reproduced: all values lie on the plane
                @test all(((p, v),) -> v ≈ g(p...), zip(result(l).points, result(l).values))
            end
        end
        # constant function: loss is the volume of the (flat) embedded simplex = simplex volume; finishes without error
        l = LearnerND((x, y) -> 1.0; bounds = ((0, 1), (0, 1)))
        run!(l, (x, y) -> 1.0, 40)
        @test loss(l) < 0.05
    end

    @testset "invariants after every tell!" begin
        for (f, bounds) in ((f2, ((-1, 1), (0, 2))), (f3, ((-1, 1), (-1, 1), (-1, 1))))
            l = LearnerND(f; bounds)
            for _ in 1:12
                ps = ask(l, 3)
                for p in ps
                    tell!(l, p, f(p...))
                    check_invariants(l)
                end
            end
            @test A.bounds_done(l)
        end
        # points told out of order and not asked
        l = LearnerND(f2; bounds = ((-1, 1), (-1, 1)))
        rng_pts = [(0.3, 0.2), (-1.0, -1.0), (0.9, -0.8), (1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (0.0, 0.0), (0.45, 0.45), (-0.5, 0.7)]
        for p in rng_pts
            tell!(l, p, f2(p...))
            check_invariants(l)
        end
        @test length(simplices(l)) >= 4
    end

    @testset "steep function samples densely, in 3D" begin
        g(x, y, z) = tanh(50x) + 0.1 * y * z
        l = LearnerND(g)
        run!(l, g, 300)
        near = count(p -> abs(p[1]) < 0.1, result(l).points) / npoints(l)
        @test near > 0.2  # a uniform grid puts 10 % into this slab
        lu = LearnerND(g; loss_per_simplex = uniform_loss_nd)
        run!(lu, g, 300)
        @test count(p -> abs(p[1]) < 0.1, result(lu).points) / npoints(lu) < 0.15
        check_invariants(l)
    end

    @testset "std_loss_nd" begin
        g(x, y, z) = tanh(50x)
        l = LearnerND(g; loss_per_simplex = std_loss_nd)
        run!(l, g, 150)
        @test count(p -> abs(p[1]) < 0.1, result(l).points) / npoints(l) > 0.2
        simplex = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        @test std_loss_nd(simplex, [0.0, 0.0, 0.0], 1.0) ≈ 0.5  # no spread: the volume
        r = sqrt(2 / 9)  # standard deviation of (0, 1, 0), population
        @test std_loss_nd(simplex, [0.0, 1.0, 0.0], 1.0) ≈ r * sqrt(0.5) + 0.5
        @test std_loss_nd(simplex, [(0.0, 0.0), (1.0, 1.0), (0.0, 0.0)], 1.0) ≈ sqrt(2) * r * sqrt(0.5) + 0.5
    end

    @testset "default and uniform loss values" begin
        tri = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        @test uniform_loss_nd(tri, [5.0, 1.0, 2.0], 1.0) ≈ 0.5
        @test default_loss_nd(tri, [0.0, 0.0, 0.0], 1.0) ≈ 0.5  # flat: simplex volume
        @test default_loss_nd(tri, [0.0, 1.0, 0.0], 1.0) ≈ sqrt(2) / 2  # area of the triangle (0,0,0), (1,0,1), (0,1,0)
        @test default_loss_nd(tri, [(0.0,), (1.0,), (0.0,)], 1.0) ≈ default_loss_nd(tri, [0.0, 1.0, 0.0], 1.0)
    end

    @testset "4D smooth function" begin
        g(a, b, c, d) = sin(2a) * cos(b) + c * d
        l = LearnerND(g; bounds = ntuple(_ -> (-1, 1), 4))
        run!(l, g, 120)
        check_invariants(l)
        @test npoints(l) == 120 && isfinite(loss(l)) && loss(l) > 0
        l1 = loss(l)
        run!(l, g, 200)
        @test loss(l) < l1
        @test length(simplices(l)) > 100
    end

    @testset "1D bounds work" begin
        g(x) = tanh(20x)
        l = LearnerND(g; bounds = ((-1, 1),))
        run!(l, g, 50; batch = 2)
        check_invariants(l)
        @test count(p -> abs(p[1]) < 0.2, result(l).points) > 15
    end

    @testset "vector values" begin
        l = LearnerND(fv; bounds = ((-1, 1), (-1, 1)), vdim = 2)
        run!(l, fv, 80)
        check_invariants(l)
        @test eltype(result(l).values) == NTuple{2, Float64}
        @test all(((p, v),) -> v == fv(p...), zip(result(l).points, result(l).values))
        @test_throws ArgumentError tell!(l, (0.123, 0.456), (1.0, 2.0, 3.0))
        @test_throws ArgumentError tell!(l, (0.123, 0.456), 1.0)
        tell!(l, (0.123, 0.456), [1.0, 2.0])  # any iterable
        @test result(l).values[end] == (1.0, 2.0)
        # the steep component is refined although the other one is smooth
        g(x, y) = (y, tanh(50x))
        lv = LearnerND(g; vdim = 2, bounds = ((-1, 1), (-1, 1)))
        run!(lv, g, 200)
        @test count(p -> abs(p[1]) < 0.1, result(lv).points) / 200 > 0.25
    end

    @testset "no duplicates with many pending" begin
        l = LearnerND(f3)
        tell_all!(l, f3, ask(l, 8))
        run!(l, f3, 20)
        first = ask(l, 60)
        second = ask(l, 60)
        all_points = vcat(first, second)
        @test length(all_points) == 120
        @test length(Set(all_points)) == 120
        @test isdisjoint(Set(all_points), Set(result(l).points))
        @test Set(pending_points(l)) == Set(all_points)
        for p in all_points  # tell them in a shuffled order, invariants hold throughout
            tell!(l, p, f3(p...))
        end
        check_invariants(l)
        @test isempty(pending_points(l))
        # the same in 2D, where pending points land on shared faces
        l2 = L2(f2)
        tell_all!(l2, f2, ask(l2, 4))
        pts = vcat([ask(l2, 25) for _ in 1:4]...)
        @test length(Set(pts)) == 100
        for p in pts
            tell!(l2, p, f2(p...))
        end
        check_invariants(l2)
    end

    @testset "pending points spread the asks" begin
        l = L2(f2)
        tell_all!(l, f2, ask(l, 4))
        run!(l, f2, 30)
        pts = ask(l, 10)  # without telling, the ten points must be spread out, not clustered
        d = minimum(norm2(a, b) for (i, a) in enumerate(pts) for b in pts[(i + 1):end])
        @test d > 1e-3
        @test loss(l; real = false) <= loss(l; real = true)
        @test loss(l; real = false) < loss(l; real = true)
    end

    @testset "ask with tell_pending = false" begin
        l = LearnerND(f3)
        tell_all!(l, f3, ask(l, 8))
        run!(l, f3, 25)
        before = (pending_points(l), result(l), loss(l))
        a = ask(l, 7; tell_pending = false)
        @test pending_points(l) == before[1] && result(l) == before[2] && loss(l) == before[3]
        @test ask(l, 7; tell_pending = false) == a
        @test ask(l, 7) == a  # the same points as with registering
        l_ref = LearnerND(f3)
        tell_all!(l_ref, f3, ask(l_ref, 8))
        run!(l_ref, f3, 25)
        @test ask(l_ref, 7; tell_pending = false) == a
        # asking without registering any number of times does not change what a later ask returns
        l2 = L2(f2)
        tell_all!(l2, f2, ask(l2, 4))
        for _ in 1:5
            ask(l2, 3; tell_pending = false)
        end
        l3 = L2(f2)
        tell_all!(l3, f2, ask(l3, 4))
        @test ask(l2, 6) == ask(l3, 6)
    end

    @testset "remove_unfinished!" begin
        l = L2(f2)
        tell_all!(l, f2, ask(l, 4))
        run!(l, f2, 20)
        first = ask(l, 5)
        @test length(pending_points(l)) == 5
        @test remove_unfinished!(l) === l
        @test isempty(pending_points(l)) && isempty(l.subtriangulations)
        @test loss(l; real = false) == loss(l)
        @test ask(l, 5) == first  # the same points are handed out again
        remove_unfinished!(l)
        tell_all!(l, f2, first)
        check_invariants(l)
        run!(l, f2, 60)
        check_invariants(l)
    end

    @testset "improvements returned with the points" begin
        l = L2(f2)
        pts, imp = A.ask_with_improvements(l, 4)
        @test all(==(Inf), imp) && length(pts) == 4
        tell_all!(l, f2, pts)
        pts, imp = A.ask_with_improvements(l, 5)
        @test all(isfinite, imp) && all(>=(0), imp)
        @test maximum(imp) ≈ loss(l) atol = 1e-8
    end

    @testset "custom loss" begin
        seen = Ref(0.0)
        scales = Ref(0.0)
        function my_loss(simplex, values, value_scale)
            seen[] = max(seen[], maximum(maximum(abs, p) for p in simplex))
            scales[] = value_scale
            @assert length(simplex) == 3 && length(values) == 3
            return 1.0 + 0.0 * sum(values)
        end
        l = LearnerND(f2; bounds = ((0, 4), (0, 2)), loss_per_simplex = my_loss)
        run!(l, f2, 30)
        @test seen[] <= 2.0 + 1e-12  # coordinates scaled by the widths (4, 2)
        @test seen[] >= 1.0
        @test scales[] > 0
        @test loss(l) == 1.0
        # a loss that is the volume reproduces uniform_loss_nd exactly
        vol(simplex, values, scale) = uniform_loss_nd(simplex, values, scale)
        a = L2(f2; loss_per_simplex = vol)
        b = L2(f2; loss_per_simplex = uniform_loss_nd)
        run!(a, f2, 40)
        run!(b, f2, 40)
        @test result(a).points == result(b).points
        # values reach the loss scaled to unit range
        vals = Ref(0.0)
        function range_loss(simplex, values, scale)
            vals[] = max(vals[], maximum(abs, values))
            return uniform_loss_nd(simplex, values, scale)
        end
        c = LearnerND((x, y) -> 1000 * x + 5; bounds = ((0, 1), (0, 1)), loss_per_simplex = range_loss)
        run!(c, (x, y) -> 1000 * x + 5, 20)
        @test 0.9 < vals[] < 1.1  # value_scale * values: the range is 1 (the offset 5 stays, scaled)
    end

    @testset "degenerate first points" begin
        l = LearnerND(f2; bounds = ((-1, 1), (-1, 1)))
        for p in [(-1.0, -1.0), (0.0, 0.0), (1.0, 1.0)]  # collinear
            tell!(l, p, f2(p...))
        end
        @test isempty(simplices(l)) && loss(l) == Inf
        @test l.tri === nothing
        ps = ask(l, 3)  # corners that are missing, then a random point
        @test length(ps) == 3 && length(Set(ps)) == 3
        tell!(l, (-1.0, 1.0), 0.5)
        check_invariants(l)
        @test length(simplices(l)) >= 1  # the fourth point breaks the degeneracy
        tell!(l, (1.0, -1.0), 0.25)
        check_invariants(l)
        @test sum(A.volumes(l.tri)) ≈ 4
        # all points in a plane in 3D, then a point out of the plane
        l3 = LearnerND(f3)
        for p in [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0), (0.0, 0.0, 0.0)]
            tell!(l3, p, f3(p...))
        end
        @test isempty(simplices(l3)) && loss(l3) == Inf
        tell!(l3, (0.0, 0.0, 1.0), 1.0)
        @test !isempty(simplices(l3))
        check_invariants(l3)
        # the first told points are not the corners
        l4 = L2(f2)
        tell_all!(l4, f2, [(0.1, 0.1), (0.5, -0.5), (-0.3, 0.6)])
        @test length(simplices(l4)) == 1 && loss(l4) == Inf  # corners are missing
        run!(l4, f2, 30)
        check_invariants(l4)
        @test isfinite(loss(l4))
    end

    @testset "2D: sample density follows the same feature as Learner2D" begin
        g(x, y) = tanh(50 * (x - 0.3))
        l = LearnerND(g; bounds = ((-1, 1), (-1, 1)))
        run!(l, g, 300)
        l2 = Learner2D(g; bounds = ((-1, 1), (-1, 1)))
        while npoints(l2) < 300
            foreach(p -> tell!(l2, p, g(p...)), ask(l2, 4))
        end
        band(l) = count(p -> abs(p[1] - 0.3) < 0.1, result(l).points) / npoints(l)
        @test band(l) > 0.3 && band(l2) > 0.3  # a uniform grid puts 10 % there
        # sample density along x agrees roughly: both put most points within 0.4 of the ridge
        wide(l) = count(p -> abs(p[1] - 0.3) < 0.4, result(l).points) / npoints(l)
        @test wide(l) > 0.6 && wide(l2) > 0.6
        # same simplex count as a Delaunay triangulation of the points (2D cross-check)
        pts = result(l).points
        @test length(simplices(l)) == length(AdaptivePort.delaunay_triangles(pts))
    end

    @testset "simplices accessor" begin
        l = LearnerND(f2; bounds = ((0, 1), (0, 1)))
        tell_all!(l, f2, ask(l, 4))
        s = simplices(l)
        @test length(s) == 2 && all(issorted, s) && issorted(s)
        @test Set(vcat(collect.(s)...)) == Set(1:4)
        tell!(l, (0.5, 0.5), 0.0)
        @test length(simplices(l)) == 4
        @test all(t -> 5 in t, simplices(l))
    end

    @testset "matches upstream" begin
        cases = (("2d_corners", f2, ((-1, 1), (-1, 1)), 1, []), ("2d_extra", f2, ((-1, 1), (-1, 1)), 1, EXTRA2),
            ("3d_extra", f3, ((-1, 1), (-1, 1), (-1, 1)), 1, EXTRA3), ("2d_vec", fv, ((-1, 1), (-1, 1)), 2, EXTRA2))
        # points agree to 1e-9; losses are rounded to 8 digits by upstream and here, so compare to 2e-8
        ptol, ltol = 1e-9, 2e-8
        for (name, f, bounds, vdim, extra) in cases
            @testset "$name" begin
                r = REF[name]
                l = LearnerND(f; bounds, vdim)
                corners = sort!(vec(collect(Iterators.product(map(b -> Float64.(b), bounds)...))))
                tell_all!(l, f, vcat(corners, [Float64.(p) for p in extra]))
                same(got, want) = length(got) == length(want) && all(zip(got, want)) do ((p, ls), (q, lq))
                    all(abs.(collect(p) .- collect(q)) .< ptol) && (ls == lq || abs(ls - lq) < ltol)
                end
                seq = []
                for _ in 1:12
                    ps, ls = A.ask_with_improvements(l, 1)
                    push!(seq, (ps[1], ls[1]))
                    tell!(l, ps[1], f(ps[1]...))
                end
                @test same(seq, r.seq)
                ps, ls = A.ask_with_improvements(l, 6)
                @test same(collect(zip(ps, ls)), r.batch)
                tell_all!(l, f, ps)
                ps, ls = A.ask_with_improvements(l, 6)
                @test same(collect(zip(ps, ls)), r.batch2)
                @test abs(loss(l) - r.loss) < ltol
                a = ask(l, 4; tell_pending = false)
                @test all(all(abs.(collect(p) .- collect(q)) .< ptol) for (p, q) in zip(a, r.nopending)) && length(a) == 4
            end
        end
    end

    @testset "runner integration" begin
        g(x, y) = tanh(10x) * y
        l = LearnerND(g; bounds = ((-1, 1), (-1, 1)))
        BlockingRunner(l; max_npoints = 60)
        @test npoints(l) >= 60 && isfinite(loss(l))
        check_invariants(l)
    end
end

norm2(a, b) = sqrt(sum(abs2, a .- b))
end # module LearnerNDTests

@testset "LearnerND" begin
    LearnerNDTests.run_tests()
end
