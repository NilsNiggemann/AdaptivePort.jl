# LearnerND on a distorted spherical shell, drawn with real 3D rendering (depth, lighting) in GLMakie.
# Left: the samples as lit spheres, size and opacity following f, with an octant cut away.
# Middle: the f = 0.5 isosurface of the learner's own Delaunay mesh (marching tetrahedra), coloured by height.
# Right: three orthogonal slices of the interpolated field.
# Run with: julia --project=examples examples/learnernd_3d.jl   (opens an interactive, rotatable window)
using AdaptivePort, GLMakie, NearestNeighbors, LinearAlgebra

const WIDTH = 0.05
const N_SAMPLES = 5000
const LEVEL = 0.5

"Radius of the shell in the direction of the point (x, y, z)."
function shell_radius(x, y, z)
    r = sqrt(x^2 + y^2 + z^2)
    phi = atan(y, x)
    cos_theta = z / max(r, 1e-12)
    return 0.6 * (1 + 0.2 * sin(3 * phi) * cos_theta + 0.15 * cos(4 * cos_theta))
end

shell(x, y, z) = exp(-((sqrt(x^2 + y^2 + z^2) - shell_radius(x, y, z)) / WIDTH)^2)

"The octant that is cut away in every panel; it faces the camera."
inside_octant(p) = p[1] < 0 && p[2] < 0 && p[3] > 0

# ponytail: stand-in for the learner's linear interpolant (LearnerND has no evaluation yet); fine for
# colouring slices, blurs features narrower than the local sample spacing. Upgrade: point location in `simplices`.
"""
    interpolate_idw(points, values, queries; k = 4)

Inverse-distance interpolation of the samples at `queries`, from the `k` nearest samples.
"""
function interpolate_idw(points, values, queries; k = 4)
    tree = KDTree(Float64[p[i] for i in 1:3, p in points])
    indices, distances = knn(tree, Float64[q[i] for i in 1:3, q in queries], k)
    return map(indices, distances) do idx, dist
        weights = 1 ./ (dist .+ 1e-6)
        sum(weights .* values[idx]) / sum(weights)
    end
end

cross3(a, b) = Point3f(a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])

"""
    isosurface(points, values, cells, level)

Triangles of the `level` set of the piecewise-linear interpolant on the tetrahedra `cells` (index tuples into
`points`), as a vector of vertices with three consecutive vertices per triangle. Normals point toward low values.
"""
function isosurface(points, values, cells, level)
    vertices = Point3f[]
    position(i) = Point3f(points[i]...)
    for cell in cells
        above = [values[i] > level for i in cell]
        n_above = count(above)
        (n_above == 0 || n_above == 4) && continue
        high = [cell[j] for j in 1:4 if above[j]]
        low = [cell[j] for j in 1:4 if !above[j]]
        # point on the edge i -- j where the interpolant crosses the level
        crossing(i, j) = position(i) + Float32((level - values[i]) / (values[j] - values[i])) * (position(j) - position(i))
        triangles = if n_above == 1 || n_above == 3
            apex, base = n_above == 1 ? (high[1], low) : (low[1], high)
            [(crossing(apex, base[1]), crossing(apex, base[2]), crossing(apex, base[3]))]
        else
            quad = (crossing(high[1], low[1]), crossing(high[1], low[2]), crossing(high[2], low[2]), crossing(high[2], low[1]))
            [(quad[1], quad[2], quad[3]), (quad[1], quad[3], quad[4])]
        end
        toward_low = sum(position, low) / length(low) - sum(position, high) / length(high)
        for (a, b, c) in triangles
            # winding such that the normal points toward low values, so lighting is consistent
            dot(cross3(b - a, c - a), toward_low) < 0 && ((b, c) = (c, b))
            push!(vertices, a, b, c)
        end
    end
    return vertices
end

"A `Axis3` without decorations, looking at the cut octant."
function cutaway_axis(position)
    ax = Axis3(position; aspect = :data, azimuth = 1.3pi, elevation = 0.2pi, perspectiveness = 0.3, limits = ((-0.9, 0.9), (-0.9, 0.9), (-0.9, 0.9)))
    hidedecorations!(ax)
    hidespines!(ax)
    return ax
end

learner = LearnerND(shell; bounds = ((-1, 1), (-1, 1), (-1, 1)))
while npoints(learner) < N_SAMPLES
    for p in ask(learner, 20)
        tell!(learner, p, shell(p...))
    end
end
data = result(learner)
points = data.points
values = collect(data.values)
cells = simplices(learner)

fig = Figure(size = (1260, 400), backgroundcolor = :white)

# left: spheres, big and opaque where f is large, tiny and faint in the background
visible_points = .!inside_octant.(points)
ax1 = cutaway_axis(fig[1, 1])
colors = [RGBAf(get(cgrad(:viridis), v), 0.25 + 0.75 * min(1, 2v)) for v in values[visible_points]]
meshscatter!(ax1, Point3f.(points[visible_points]); color = colors, markersize = Float32.(0.008 .+ 0.02 .* values[visible_points]),
             transparency = true, specular = 0.8f0, shininess = 60f0)
Label(fig[2, 1], "$(npoints(learner)) samples, octant cut"; color = :gray30, tellwidth = false)

# middle: isosurface straight from the adaptive tetrahedra
vertices = isosurface(points, values, cells, LEVEL)
triangle_centroid(t) = (vertices[3t - 2] + vertices[3t - 1] + vertices[3t]) / 3
kept = [t for t in 1:length(vertices) ÷ 3 if !inside_octant(triangle_centroid(t))]
kept_vertices = vertices[[3t - k for t in kept for k in (2, 1, 0)]]
faces = collect(reshape(1:3length(kept), 3, :)')   # triangle i = vertices 3i - 2, 3i - 1, 3i
ax2 = cutaway_axis(fig[1, 2])
mesh!(ax2, kept_vertices, faces; color = [v[3] for v in kept_vertices], colormap = :turbo, shading = true)
Label(fig[2, 2], "f = $LEVEL isosurface of the sampling mesh, $(length(kept)) triangles"; color = :gray30, tellwidth = false)

# right: three orthogonal slices, each missing the cut octant
us = range(-1, 1; length = 160)
ax3 = cutaway_axis(fig[1, 3])
planes = (((u, v) -> (0.0, u, v)) => (p -> !(p[2] < 0 && p[3] > 0)),
          ((u, v) -> (u, 0.0, v)) => (p -> !(p[1] < 0 && p[3] > 0)),
          ((u, v) -> (u, v, 0.0)) => (p -> !(p[1] < 0 && p[2] < 0)))
for (plane, shown) in planes
    xyz = [plane(u, v) for u in us, v in us]
    slice = reshape(interpolate_idw(points, values, vec(Point3f.(xyz))), size(xyz))
    slice[.!shown.(xyz)] .= NaN
    surface!(ax3, getindex.(xyz, 1), getindex.(xyz, 2), getindex.(xyz, 3); color = slice, shading = NoShading,
             colormap = :viridis, colorrange = (0, 1), nan_color = :transparent)
end
Label(fig[2, 3], "orthogonal slices"; color = :gray30, tellwidth = false)

rowgap!(fig.layout, 4)
fig

# save("learnernd_3d.png", fig; px_per_unit = 2)
