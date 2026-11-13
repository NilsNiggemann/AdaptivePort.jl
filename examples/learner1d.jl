# Adaptive sampling of a 1D function with Learner1D.
# Run with: julia --project=. examples/learner1d.jl
using AdaptivePort

f(x) = sin(8x) + 0.2cos(23x)

learner = Learner1D(f; bounds=(-1, 1))

while !done(learner; tol=0.05) && npoints(learner) < 1000
    points = ask(learner, 1)
    isempty(points) && break
    for x in points
        tell!(learner, x, f(x))
    end
end

data = result(learner)
println("samples: ", npoints(learner))
println("final loss: ", loss(learner))
for (x, y) in zip(data.x, data.y)
    println(round(x; digits=4), "  ", round(y; digits=4))
end
