# N-dimensional incremental Delaunay triangulation, the geometric layer of LearnerND.
# A Julia port of `adaptive/learner/triangulation.py` from Python Adaptive.
#
# BSD 3-Clause License
#
# Copyright (c) 2017-2023, Adaptive authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
    TriangulationND{N, T, S}

Delaunay triangulation of points in `N` dimensions, built incrementally by the Bowyer-Watson algorithm; a port of upstream's `Triangulation`. Unexported.

Fields: `vertices` (the points, `NTuple{N, T}`), `simplices` (the simplices, each a sorted `S = NTuple{N + 1, Int}` of indices into `vertices`) and `vertex_to_simplices` (for every vertex the simplices containing it).

Invariants (checked by the tests after every insertion):
- Every simplex is listed at each of its vertices and nowhere else.
- A face (`N` vertices) belongs to at most two simplices; faces with one simplex form the hull.
- The simplices have disjoint interiors and, up to the tolerances below, empty circumspheres.

Vertices of the input that are not (yet) part of a simplex have an empty `vertex_to_simplices` entry. Construct with `TriangulationND(points)` and extend with [`add_point!`](@ref AdaptivePort.add_point!).

Deviations from upstream: dimension 1 is supported; the initial triangulation of more than `N + 1` points starts from one non-degenerate simplex and inserts the others (upstream calls `scipy.spatial.Delaunay`); the rank test uses a relative tolerance of `1e-10`.
"""
struct TriangulationND{N, T <: AbstractFloat, S <: Tuple}
    vertices::Vector{NTuple{N, T}}
    simplices::Set{S}
    vertex_to_simplices::Vector{Set{S}}
end

const SIMPLEX_EPS = 1e-8

# Simplex of `N + 1` vertices of the triangulation, sorted.
sorted_simplex(s) = Tuple(sort!(collect(s)))

# The `N + 1` faces (vertex tuples with one vertex left out) of the sorted simplex `s`.
faces_of(s::NTuple{M, Int}) where {M} = ntuple(i -> ntuple(j -> s[j < i ? j : j + 1], M - 1), M)

dimension(::TriangulationND{N}) where {N} = N

function add_simplex!(tri::TriangulationND, simplex)
    s = sorted_simplex(simplex)
    push!(tri.simplices, s)
    for v in s
        push!(tri.vertex_to_simplices[v], s)
    end
    return s
end

function delete_simplex!(tri::TriangulationND, simplex)
    s = sorted_simplex(simplex)
    delete!(tri.simplices, s)
    for v in s
        delete!(tri.vertex_to_simplices[v], s)
    end
    return s
end

Base.copy(tri::TriangulationND) = TriangulationND(copy(tri.vertices), copy(tri.simplices), [copy(s) for s in tri.vertex_to_simplices])

# Matrix whose columns are the edge vectors `pts[k] - pts[1]`, `k = 2, ...`.
function edge_matrix(pts)
    T = eltype(pts[1])
    V = Matrix{T}(undef, length(pts[1]), length(pts) - 1)
    for k in 2:length(pts), i in axes(V, 1)
        V[i, k - 1] = pts[k][i] - pts[1][i]
    end
    return V
end

"""
    simplex_volume_in_embedding(vertices)

Volume of the simplex spanned by `vertices` (a `Vector` of `k + 1` points of `NTuple{D}`, `k <= D`), also when `k < D` (for example the area of a triangle in 3D). Equals `sqrt(det(V' * V)) / k!` for the matrix `V` of edge vectors.

Upstream uses Heron's formula (`D = 2`) or the Cayley-Menger determinant; the Gram determinant gives the same value. A slightly negative squared volume from rounding is clamped to zero (upstream raises beyond `-1e-15`).
"""
function simplex_volume_in_embedding(vertices)
    V = edge_matrix(vertices)
    k = size(V, 2)
    vol = size(V, 1) == k ? abs(LinearAlgebra.det(V)) : sqrt(max(LinearAlgebra.det(V' * V), zero(eltype(V))))
    return vol / factorial(k)
end

# Volume of the `N`-simplex with `N + 1` vertices `pts` in `N` dimensions.
volume(pts) = simplex_volume_in_embedding(pts)

volume(tri::TriangulationND, simplex) = volume(view(tri.vertices, collect(simplex)))
volumes(tri::TriangulationND) = [volume(tri, s) for s in tri.simplices]

# Apply `transform` (an `N x N` matrix, or `nothing` for the identity) to the point `p` as a row vector, as upstream's `dot(p, transform)`.
apply_transform(p, ::Nothing) = p
apply_transform(p::NTuple{N}, M::AbstractMatrix) where {N} = ntuple(j -> sum(p[i] * M[i, j] for i in 1:N), N)

"""
    circumsphere(pts)

Centre (`NTuple`) and radius of the sphere through the `N + 1` points `pts` in `N` dimensions. Singular input (a flat simplex) throws `LinearAlgebra.SingularException` or returns non-finite numbers.

Solves `2 (p_i - p_1) ⋅ (c - p_1) = |p_i - p_1|^2` for the centre `c`; upstream uses closed-form determinants for `N = 2, 3` and a Cayley-Menger-like determinant otherwise.
"""
function circumsphere(pts)
    p0 = pts[1]
    N = length(p0)
    A = 2 .* transpose(edge_matrix(pts))
    b = [sum(abs2, pts[i + 1] .- p0) for i in 1:N]
    x = A \ b
    return ntuple(i -> p0[i] + x[i], N), sqrt(sum(abs2, x))
end

# Sign (-1, 0, 1) of the orientation of the `N` points `face` seen from `origin`; 0 if (nearly) in one hyperplane.
function orientation(face, origin)
    N = length(origin)
    M = [face[i][j] - origin[j] for i in 1:N, j in 1:N]
    logdet, sign = LinearAlgebra.logabsdet(M)
    return logdet < -50 ? 0 : Int(sign)
end

# Barycentric coordinates `alpha` of `point` with respect to `simplex_pts` (all but the first vertex).
barycentric(point, pts) = edge_matrix(pts) \ collect(point .- pts[1])

function point_in_simplex(point, pts; eps = SIMPLEX_EPS)
    alpha = barycentric(point, pts)
    return all(>(-eps), alpha) && sum(alpha) < 1 + eps
end

point_in_simplex(tri::TriangulationND, point, simplex; eps = SIMPLEX_EPS) = point_in_simplex(point, tri.vertices[collect(simplex)]; eps)

"""
    TriangulationND(points)

Triangulation of the distinct `points` (`Vector` of `NTuple{N}`). Throws `ArgumentError` for fewer than `N + 1` points, for points in one hyperplane, and for coinciding points.

The vertex indices follow the order of `points`.
"""
function TriangulationND(points::AbstractVector{<:NTuple{N, Real}}) where {N}
    length(points) >= N + 1 || throw(ArgumentError("need at least $(N + 1) points in $N dimensions, got $(length(points))"))
    T = float(reduce(promote_type, fieldtypes(eltype(points))))
    vertices = [NTuple{N, T}(p) for p in points]
    chosen = [1]
    for i in 2:length(vertices)
        length(chosen) == N + 1 && break
        E = edge_matrix(vertices[vcat(chosen, i)])
        LinearAlgebra.rank(E; rtol = 1e-10) == length(chosen) && push!(chosen, i)
    end
    length(chosen) == N + 1 || throw(ArgumentError("the points lie in one hyperplane, the initial simplex has zero volume"))
    S = NTuple{N + 1, Int}
    tri = TriangulationND(vertices, Set{S}(), [Set{S}() for _ in vertices])
    add_simplex!(tri, chosen)
    for i in eachindex(vertices)
        i in chosen || insert_vertex!(tri, i)
    end
    return tri
end

# Simplices that share a whole face with `simplex` among the simplices in `candidates`.
face_sharing(candidates, simplex, N) = [s for s in candidates if count(in(simplex), s) == N]

"""
    containing(tri, face)

Simplices of `tri` that contain all vertices of `face` (a tuple of vertex indices).
"""
containing(tri::TriangulationND, face) = intersect((tri.vertex_to_simplices[v] for v in face)...)

# The simplices sharing a face with `simplex`.
neighbors(tri::TriangulationND{N}, simplex) where {N} = Set(s for face in faces_of(simplex) for s in containing(tri, face) if s != simplex)

# Multiplicity of every face of `simplices`.
function face_counts(simplices)
    counts = Dict{Any, Int}()
    for s in simplices, f in faces_of(s)
        counts[f] = get(counts, f, 0) + 1
    end
    return counts
end

faces(tri::TriangulationND) = keys(face_counts(tri.simplices))

# Vertices on the hull: those of faces that belong to one simplex only. Throws if a face has more than two simplices.
function hull(tri::TriangulationND)
    counts = face_counts(tri.simplices)
    any(>(2), values(counts)) && error("broken triangulation: a face belongs to more than two simplices")
    return Set(v for (f, c) in counts if c == 1 for v in f)
end

# Volume of a simplex divided by the mean absolute edge-vector entry to the power `N`: depends on the shape only, used to detect flat simplices.
function relative_volume(tri::TriangulationND{N}, simplex) where {N}
    V = edge_matrix(tri.vertices[collect(simplex)])
    return volume(tri, simplex) / (sum(abs, V) / length(V))^N
end

is_almost_flat(tri::TriangulationND, simplex) = relative_volume(tri, simplex) < SIMPLEX_EPS

function in_circumsphere(tri::TriangulationND, pt_index, simplex, transform)
    center, radius = circumsphere([apply_transform(tri.vertices[v], transform) for v in simplex])
    pt = apply_transform(tri.vertices[pt_index], transform)
    return sqrt(sum(abs2, center .- pt)) < radius * (1 + SIMPLEX_EPS)
end

# Vertices of `simplex` whose barycentric coordinate for `point` is nonzero (the face containing the point); empty if outside.
function reduced_simplex(tri::TriangulationND{N}, point, simplex; eps = SIMPLEX_EPS) where {N}
    alpha = barycentric(point, tri.vertices[collect(simplex)])
    (any(<(-eps), alpha) || sum(alpha) > 1 + eps) && return Int[]
    result = [simplex[i + 1] for i in 1:N if alpha[i] > eps]
    sum(alpha) < 1 - eps && pushfirst!(result, simplex[1])
    return result
end

# First simplex containing `point`, or `nothing` outside the hull. O(number of simplices).
function locate_point(tri::TriangulationND, point)
    for s in tri.simplices
        point_in_simplex(tri, point, s) && return s
    end
    return nothing
end

# Add simplices joining the vertex `pt_index` to every hull face that faces it; returns them.
function extend_hull!(tri::TriangulationND{N}, pt_index) where {N}
    hull_faces = sort!([f for (f, c) in face_counts(tri.simplices) if c == 1])
    hull_points = tri.vertices[sort!(collect(Set(v for f in hull_faces for v in f)))]
    center = ntuple(i -> sum(p[i] for p in hull_points) / length(hull_points), N)
    new_vertex = tri.vertices[pt_index]
    new_simplices = Set{eltype(tri.simplices)}()
    for face in hull_faces
        pts = tri.vertices[collect(face)]
        # new vertex on the far side of the face from the centre: add the simplex
        if orientation(pts, center) == -orientation(pts, new_vertex)
            simplex = sorted_simplex((face..., pt_index))
            if !is_almost_flat(tri, simplex)
                add_simplex!(tri, simplex)
                push!(new_simplices, simplex)
            end
        end
    end
    isempty(new_simplices) && throw(ArgumentError("candidate vertex is inside the hull"))
    return new_simplices
end

# Create a hole around vertex `pt_index` (all simplices whose circumsphere contains it) and retriangulate it. Returns `(deleted, added)`.
function bowyer_watson!(tri::TriangulationND{N, T, S}, pt_index; containing = nothing, transform = nothing) where {N, T, S}
    queue = Set{S}()
    done = Set{S}()
    bad = Set{S}()
    containing === nothing ? union!(queue, tri.vertex_to_simplices[pt_index]) : push!(queue, containing)
    while !isempty(queue)
        simplex = pop!(queue)
        push!(done, simplex)
        in_circumsphere(tri, pt_index, simplex, transform) || continue
        delete_simplex!(tri, simplex)
        push!(bad, simplex)
        candidates = setdiff(union((tri.vertex_to_simplices[v] for v in simplex)...), done)
        union!(queue, face_sharing(candidates, simplex, N))
    end
    for (face, c) in sort!(collect(face_counts(bad)))
        c < 2 && !(pt_index in face) || continue
        simplex = sorted_simplex((face..., pt_index))
        is_almost_flat(tri, simplex) || add_simplex!(tri, simplex)
    end
    new = copy(tri.vertex_to_simplices[pt_index])
    deleted, added = setdiff(bad, new), setdiff(new, bad)
    old_vol = sum(s -> volume(tri, s), deleted; init = 0.0)
    new_vol = sum(s -> volume(tri, s), added; init = 0.0)
    isapprox(old_vol, new_vol; rtol = 1e-5, atol = 1e-8) || error("volume not conserved by the insertion: $old_vol != $new_vol")
    return deleted, added
end

# Insert the already stored vertex `pt_index` into the triangulation. Returns `(deleted, added)` simplices.
function insert_vertex!(tri::TriangulationND, pt_index; simplex = nothing, transform = nothing)
    point = tri.vertices[pt_index]
    simplex === nothing && (simplex = locate_point(tri, point))
    if simplex === nothing  # outside the hull
        temporary = extend_hull!(tri, pt_index)
        deleted, added = bowyer_watson!(tri, pt_index; transform)
        return setdiff(deleted, temporary), union(added, setdiff(temporary, deleted))
    end
    reduced = reduced_simplex(tri, point, simplex)
    isempty(reduced) && throw(ArgumentError("point $point lies outside the given simplex"))
    length(reduced) == 1 && throw(ArgumentError("point $point is already a vertex of the triangulation"))
    return bowyer_watson!(tri, pt_index; containing = simplex, transform)
end

"""
    add_point!(tri::TriangulationND, point; simplex = nothing, transform = nothing)

Insert `point` into `tri` (Bowyer-Watson inside the hull, hull extension plus Bowyer-Watson outside) and return `(deleted, added)`, the sets of simplices removed and created. The new vertex gets the next index.

**Keyword Arguments**
- `simplex`: a simplex containing `point`; saves the O(number of simplices) point location.
- `transform`: `N x N` matrix applied to the vertices (as row vectors) when testing circumspheres, so that Delaunay's criterion holds in the transformed (for example scaled) space.

Throws `ArgumentError`, leaving `tri` unchanged, if `point` is already a vertex or `simplex` does not contain it.
"""
function add_point!(tri::TriangulationND{N, T}, point; simplex = nothing, transform = nothing) where {N, T}
    push!(tri.vertices, NTuple{N, T}(point))
    push!(tri.vertex_to_simplices, empty(first(tri.vertex_to_simplices)))
    try
        return insert_vertex!(tri, length(tri.vertices); simplex, transform)
    catch
        pop!(tri.vertices)
        pop!(tri.vertex_to_simplices)
        rethrow()
    end
end
