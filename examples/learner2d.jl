# Adaptive sampling of a 2D function with Learner2D.
# Run with: julia --project=. examples/learner2d.jl
using AdaptivePort

f(x, y) = sin(5x) * cos(7y) + 0.3 * exp(-20 * ((x - 0.3)^2 + (y + 0.2)^2))

learner = Learner2D(f; bounds=((-1, 1), (-0.75, 0.75)))

while !done(learner; tol=0.01) && npoints(learner) < 2000
    points = ask(learner, 5)
    isempty(points) && break
    for (x, y) in points
        tell!(learner, (x, y), f(x, y))
    end
end

data = result(learner)
println("samples: ", npoints(learner))
println("final loss: ", loss(learner))
println("triangles: ", length(triangles(learner)))

# Adaptive, not a grid: triangle areas span orders of magnitude.
pts = data.points
areas = [abs((pts[j][1] - pts[i][1]) * (pts[k][2] - pts[i][2]) - (pts[k][1] - pts[i][1]) * (pts[j][2] - pts[i][2])) / 2 for (i, j, k) in triangles(learner)]
println("triangle area: min ", minimum(areas), ", max ", maximum(areas), ", ratio ", round(Int, maximum(areas) / minimum(areas)))
for i in 1:10:length(pts)
    (x, y), z = pts[i], data.values[i]
    println(round(x; digits=4), "  ", round(y; digits=4), "  ", round(z; digits=4))
end
