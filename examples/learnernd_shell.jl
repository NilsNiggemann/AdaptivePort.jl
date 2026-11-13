# LearnerND on a thin, sinusoidally distorted spherical shell: the adaptive samples as a 3D scatter coloured by f.
# Left: a thin slab through the middle; right: a cutaway whose cut planes show the nearest-sample (Voronoi) value.
# Run with: julia --project=examples examples/learnernd_shell.jl
using AdaptivePort, CairoMakie, NearestNeighbors

const WIDTH = 0.05

"Radius of the shell in the direction of the point (x, y, z)."
function shell_radius(x, y, z)
    r = sqrt(x^2 + y^2 + z^2)
    phi = atan(y, x)
    cos_theta = z / max(r, 1e-12)
    return 0.6 * (1 + 0.2 * sin(3 * phi) * cos_theta + 0.15 * cos(4 * cos_theta))
end

shell(x, y, z) = exp(-((sqrt(x^2 + y^2 + z^2) - shell_radius(x, y, z)) / WIDTH)^2)

learner = LearnerND(shell; bounds = ((-1, 1), (-1, 1), (-1, 1)))
while npoints(learner) < 5000
    for p in ask(learner, 20)
        tell!(learner, p, shell(p...))
    end
end
data = result(learner)
values = collect(data.values)
points = Point3f.(data.points)

markerstyle = (; colormap = :viridis, colorrange = (0, 1), strokewidth = 0.6, strokecolor = (:black, 0.7))
fig = Figure(size = (820, 420), backgroundcolor = :white)

# left: thin slab |z| < 0.1 through the middle, background points included
slab = [p[3] for p in data.points] .|> abs .< 0.1
ax1 = Axis(fig[1, 1]; aspect = DataAspect())
hidedecorations!(ax1)
hidespines!(ax1)
scatter!(ax1, [Point2f(p[1], p[2]) for p in data.points[slab]]; color = values[slab], markersize = 9, markerstyle...)
Label(fig[2, 1], "slab |z| < 0.1, $(count(slab)) points"; color = :gray30, tellwidth = false)

# right: the octant x, y < 0, z > 0 (facing the camera) is cut open; its three bounding planes show the
# nearest-sample (Voronoi) value of the data
tree = KDTree(Float64[p[i] for i in 1:3, p in data.points])
function voronoi_cut(xyz)
    indices, _ = knn(tree, Float64[p[i] for i in 1:3, p in vec(xyz)], 1)
    return reshape(values[first.(indices)], size(xyz))
end
inside_octant(p) = p[1] < 0 && p[2] < 0 && p[3] > 0
keep = (values .> 0.02) .& .!inside_octant.(data.points)
ax2 = Axis3(fig[1, 2]; aspect = :data, azimuth = 1.3pi, elevation = 0.2pi, perspectiveness = 0.3)
hidedecorations!(ax2)
hidespines!(ax2)
scatter!(ax2, points[keep]; color = values[keep], markersize = 7, markerstyle...)
us = range(-1, 0; length = 200)
planes = ((u, v) -> (0.0, u, -v), (u, v) -> (u, 0.0, -v), (u, v) -> (u, v, 0.0))
for plane in planes
    xyz = [plane(u, v) for u in us, v in us]
    surface!(ax2, getindex.(xyz, 1), getindex.(xyz, 2), getindex.(xyz, 3); color = voronoi_cut(xyz), shading = NoShading, colormap = :viridis, colorrange = (0, 1))
end
Label(fig[2, 2], "cutaway with Voronoi cuts"; color = :gray30, tellwidth = false)
fig

# save("learnernd_shell.png", fig; px_per_unit = 2)
