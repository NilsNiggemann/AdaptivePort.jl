# Learner2D on the neutron structure factor S(q) of nearest-neighbour classical spin ice in the
# self-consistent Gaussian approximation (SCGA) on the pyrochlore lattice, in the (h, h, l) plane.
# S(q) has pinch points (bow-ties) at the bcc reciprocal-lattice points; the adaptive samples gather there.
# Shown: Voronoi cells of the samples (clipped to the box) with the samples overlaid.
# q = (h, k, l) is in units of 2pi / a, a = cubic lattice constant.
# Run with: julia --project=examples examples/learner2d_spinice.jl
using AdaptivePort, CairoMakie, MakieHelpers, LinearAlgebra
import DelaunayTriangulation as DT

const T = 0.001  # temperature in units of the Ising coupling, i.e. of the effective nn coupling
# sublattice positions in the cubic cell and local <111> Ising axes of the four sublattices
const R = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]] ./ 4
const Z = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]] ./ sqrt(3)
# primitive vectors of the reciprocal (bcc) lattice
const B = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]

"Ising coupling matrix of the antiferromagnetic nn bonds in the atomic-position gauge."
function coupling(q)
    return [a == b ? 0.0 : 2cos(2pi * dot(q, R[a] .- R[b])) for a in 1:4, b in 1:4]
end

"Lagrange multiplier `lambda` fixing the spin length: the mean of `tr(G)/4` over the Brillouin zone is 1."
function spin_length_multiplier(beta; n = 24)
    us = ((1:n) .- 0.5) ./ n
    eigs = reduce(vcat, [eigvals(Symmetric(coupling(sum(u[i] .* B[i] for i in 1:3)))) for u in Iterators.product(us, us, us)])
    mean_G(lambda) = sum(1 ./ (lambda .+ beta .* eigs)) / length(eigs)
    lo, hi = 2beta, 2beta + 20   # the flat band at -2 bounds lambda from below
    for _ in 1:60
        mid = (lo + hi) / 2
        mean_G(mid) > 1 ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

const BETA = 1 / T
const LAMBDA = spin_length_multiplier(BETA)

"Neutron structure factor sum_ab G_ab(q) z_a . (1 - q q^T / q^2) . z_b."
function structure_factor(h, k, l)
    q = [h, k, l]
    G = inv(Symmetric(LAMBDA * I + BETA * coupling(q)))
    q2 = max(dot(q, q), 1e-9)
    return sum(G[a, b] * (dot(Z[a], Z[b]) - dot(Z[a], q) * dot(Z[b], q) / q2) for a in 1:4, b in 1:4)
end

"Structure factor in the (h, h, l) plane, i.e. q_x = q_y."
hhl(h, l) = structure_factor(h, h, l)

learner = Learner2D(hhl; bounds = ((-3, 3), (-3, 3)))
while npoints(learner) < 2000
    for p in ask(learner, 20)
        tell!(learner, p, hhl(p...))
    end
end
data = result(learner)
pts = data.points
crange = (0, sort(data.values)[round(Int, 0.98 * length(data.values))])

vorn = DT.voronoi(AdaptivePort.delaunay(pts); clip = true)
cells = [Point2f.(DT.get_polygon_coordinates(vorn, j)) for j in eachindex(pts)]

with_theme(theme_SimpleTicks(; publication = true)) do
    fig = Figure(size = (420, 380))
    ax = Axis(fig[1, 1]; xlabel = L"h = k", ylabel = L"l", aspect = 1)
    poly!(ax, cells; color = data.values, colormap = :viridis, colorrange = crange)
    scatter!(ax, Point2f.(pts); color = :white, markersize = 1.5)
    limits!(ax, -3, 3, -3, 3)  # no autolimit margin
    Colorbar(fig[1, 2]; colormap = :viridis, colorrange = crange, label = L"S(\mathbf{q})")
    tickcolor!(ax, :white)
    save(joinpath(@__DIR__, "..", "docs", "src", "assets", "learner2d_spinice.png"), fig; px_per_unit = 1)
    fig
end
