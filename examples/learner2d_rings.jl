# Learner2D on two thin rings: an elliptic one and one with radius 0.55(1 + 0.35 sin(4 phi)).
# Top: naive uniform grid; bottom: adaptive samples (Voronoi cells of the points, clipped to the
# convex hull, points overlaid). Same number of evaluations in both rows.
# Run with: julia --project=examples examples/learner2d_rings.jl
using AdaptivePort, CairoMakie, MakieHelpers
import DelaunayTriangulation as DT

const w = 0.04  # ring width
ellipse(x, y) = exp(-((hypot(x / 1.2, y) - 0.6) / w)^2)
sine_ring(x, y) = exp(-((hypot(x, y) - 0.55 * (1 + 0.35 * sin(4 * atan(y, x)))) / w)^2)

function sample(f; budget = 500)
    learner = Learner2D(f; bounds = ((-1.0, 1.0), (-1.0, 1.0)))
    while npoints(learner) < budget
        for p in ask(learner, 10)
            tell!(learner, p, f(p...))
        end
    end
    return result(learner)
end

budget = 800
n = round(Int, sqrt(budget))
h = 2 / n
centers = range(-1 + h / 2, 1 - h / 2; length = n)  # cells tile [-1, 1]^2 exactly

with_theme(theme_SimpleTicks(; publication = true)) do
    fig = Figure(size = (680, 600))
    axs = Axis[]
    for (i, f) in enumerate((ellipse, sine_ring))
        ax = Axis(fig[1, i]; ylabel = L"y", aspect = 1, xticklabelsvisible = false, yticklabelsvisible = i == 1)
        heatmap!(ax, centers, centers, f.(centers, centers'); colormap = :viridis, colorrange = (0, 1))
        push!(axs, ax)

        data = sample(f; budget = n^2)
        pts = data.points
        vorn = DT.voronoi(AdaptivePort.delaunay(pts); clip = true)
        cells = [Point2f.(DT.get_polygon_coordinates(vorn, j)) for j in eachindex(pts)]
        ax = Axis(fig[2, i]; xlabel = L"x", ylabel = L"y", aspect = 1, yticklabelsvisible = i == 1)
        poly!(ax, cells; color = data.values, colormap = :viridis, colorrange = (0, 1))
        scatter!(ax, Point2f.(pts); color = :white, markersize = 2)
        limits!(ax, -1, 1, -1, 1)  # no autolimit margin
        push!(axs, ax)
    end
    Colorbar(fig[1:2, 3]; colormap = :viridis, colorrange = (0, 1), label = L"f(x, y)")
    tickcolor!.(axs, :white)
    panellabels!(axs[[1, 3, 2, 4]]; color = :white)  # row-major: (a)(b) naive, (c)(d) adaptive
    rowgap!(fig.layout, 8)
    save(joinpath(@__DIR__, "..", "docs", "src", "assets", "learner2d_rings.png"), fig; px_per_unit = 1)
    fig
end
