# Geometric layer of Learner2D. This is the only file that touches DelaunayTriangulation.jl;
# triangles leave it as counter-clockwise `NTuple{3,Int}` index triples into the input points.

"""
    delaunay(points)

Delaunay triangulation of the distinct 2D `points` (a `Vector` of `NTuple{2}`), as a `DelaunayTriangulation.Triangulation`.

Insertion order is randomised with a fixed seed (fast point location, deterministic result).
"""
delaunay(points) = DT.triangulate(points; rng = Random.Xoshiro(0))

"""
    delaunay_triangles(points)

Counter-clockwise index triples of the non-degenerate Delaunay triangles of `points`, each rotated to start at its smallest index and sorted, so the result is deterministic. Empty for fewer than three points.

DelaunayTriangulation.jl returns one zero-area triangle for collinear input; such triangles are dropped.
"""
function delaunay_triangles(points)
    length(points) < 3 && return NTuple{3,Int}[]
    triangles = NTuple{3,Int}[]
    for t in DT.each_solid_triangle(delaunay(points))
        i, j, k = DT.triangle_vertices(t)
        signed_area(points[i], points[j], points[k]) > 0 || continue
        m = min(i, j, k)
        push!(triangles, m == i ? (i, j, k) : m == j ? (j, k, i) : (k, i, j))
    end
    return sort!(triangles)
end

"""
    signed_area(a, b, c)

Signed area of the triangle `(a, b, c)`, positive if counter-clockwise.
"""
signed_area(a, b, c) = ((b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1])) / 2

"""
    interpolate_linear(tri, values, q)

Piecewise-linear interpolant at `q` of `values` (one per point of the triangulation `tri`); `NaN` outside the convex hull.
"""
function interpolate_linear(tri, values, q)
    i, j, k = DT.triangle_vertices(DT.find_triangle(tri, q))
    min(i, j, k) < 1 && return oftype(float(values[1]), NaN)  # ghost triangle: q outside the hull
    pts = DT.get_points(tri)
    a, b, c = pts[i], pts[j], pts[k]
    A = signed_area(a, b, c)
    return (signed_area(q, b, c) * values[i] + signed_area(a, q, c) * values[j] + signed_area(a, b, q) * values[k]) / A
end
