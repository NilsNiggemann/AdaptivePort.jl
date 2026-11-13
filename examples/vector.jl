# Vector-valued functions with Learner1D and Learner2D (vdim = 2): both components share one set of points.
# Run with: julia --project=. examples/vector.jl
using AdaptivePort

# 1D: a smooth component and a steep step at x = 0.3. The points follow the step, the value scale is the larger range.
f1(x) = (sin(3x), tanh(60 * (x - 0.3)))
learner = Learner1D(f1; bounds = (-1, 1), vdim = 2)
while !done(learner; tol = 0.01) && npoints(learner) < 400
    for x in ask(learner, 4)
        tell!(learner, x, f1(x))
    end
end
data = result(learner)
near_step = count(x -> abs(x - 0.3) < 0.1, data.x)
println("Learner1D: ", npoints(learner), " points, loss ", round(loss(learner); digits = 4))
println("  fraction within 0.1 of the step: ", round(near_step / npoints(learner); digits = 3), " (uniform: 0.1)")
println("  first value: ", data.y[1])

# 2D: a smooth component and a thin ring in the second one.
f2(x, y) = (x * y, tanh(40 * (sqrt(x^2 + y^2) - 0.5)))
learner2 = Learner2D(f2; bounds = ((-1, 1), (-1, 1)), vdim = 2)
while !done(learner2; tol = 0.01) && npoints(learner2) < 1500
    for p in ask(learner2, 10)
        tell!(learner2, p, f2(p...))
    end
end
data2 = result(learner2)
near_ring = count(p -> abs(sqrt(p[1]^2 + p[2]^2) - 0.5) < 0.1, data2.points)
println("Learner2D: ", npoints(learner2), " points, loss ", round(loss(learner2); digits = 4))
println("  fraction within 0.1 of the ring: ", round(near_ring / npoints(learner2); digits = 3), " (uniform: 0.25)")
println("  first value: ", data2.values[1])
