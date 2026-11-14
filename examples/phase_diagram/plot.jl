# Plot the phase diagram from phase_diagram.csv (written by run_distributed.jl or run_mpi.jl). Without that
# file the data are computed here, in this process.
# Run with: julia --project=examples examples/phase_diagram/plot.jl
using AdaptivePort, CairoMakie, MakieHelpers
import DelaunayTriangulation as DT

include(joinpath(@__DIR__, "model.jl"))

path = joinpath(@__DIR__, "phase_diagram.csv")
learner = Learner2D(order_parameters; bounds = ((0.0, 1.0), (0.0, 1.0)), vdim = 2)
if isfile(path)
    load_data!(learner, path)
else
    BlockingRunner(learner; max_npoints = 1500)
end

data = result(learner)
pts = data.points
contrast = [mA - mB for (mA, mB) in data.values]  # > 0 in phase A, < 0 in phase B, 0 disordered
vorn = DT.voronoi(AdaptivePort.delaunay(pts); clip = true)
cells = [Point2f.(DT.get_polygon_coordinates(vorn, j)) for j in eachindex(pts)]

with_theme(theme_SimpleTicks(; publication = true)) do
    fig = Figure(size = (680, 300))
    ax1 = Axis(fig[1, 1]; xlabel = L"g", ylabel = L"T", aspect = 1)
    poly!(ax1, cells; color = contrast, colormap = :bwr, colorrange = (-1, 1))
    text!(ax1, [0.12, 0.88, 0.5], [0.25, 0.25, 0.8]; text = [L"\textrm{A}$$", L"\textrm{B}$$", L"\textrm{disordered}$$"], align = (:center, :center))
    ax2 = Axis(fig[1, 2]; xlabel = L"g", aspect = 1, yticklabelsvisible = false)
    scatter!(ax2, Point2f.(pts); color = :black, markersize = 1.5)
    linkaxes!(ax1, ax2)
    limits!(ax1, 0, 1, 0, 1)  # no autolimit margin
    Colorbar(fig[1, 3]; colormap = :bwr, colorrange = (-1, 1), label = L"m_A - m_B")
    panellabels!([ax1, ax2])
    save(joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "phase_diagram.png"), fig; px_per_unit = 1)
    fig
end
