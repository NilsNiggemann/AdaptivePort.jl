# Adaptive sampling of a 3D function with LearnerND: a thin spherical shell and a smooth background.
# Run with: julia --project=. examples/learnernd.jl
using AdaptivePort

f(x, y, z) = tanh(30 * (sqrt(x^2 + y^2 + z^2) - 0.6)) + 0.2 * sin(3x) * y

learner = LearnerND(f; bounds=((-1, 1), (-1, 1), (-1, 1)))

while npoints(learner) < 600
    points = ask(learner, 6)
    for p in points
        tell!(learner, p, f(p...))
    end
end

data = result(learner)
println("samples: ", npoints(learner))
println("final loss: ", loss(learner))
println("simplices: ", length(simplices(learner)))

# Adaptive, not a grid: a uniform sample puts about 15 % of the points within 0.1 of the shell (shell volume / box volume).
near_shell(p) = abs(sqrt(p[1]^2 + p[2]^2 + p[3]^2) - 0.6) < 0.1
println("fraction of samples within 0.1 of the shell: ", round(count(near_shell, data.points) / npoints(learner); digits=3))

# Vector-valued functions: `vdim` components, stored as tuples.
g(x, y) = (sin(5x) + y, tanh(20 * (x - y)))
vector_learner = LearnerND(g; bounds=((-1, 1), (-1, 1)), vdim=2)
while npoints(vector_learner) < 200
    for p in ask(vector_learner, 4)
        tell!(vector_learner, p, g(p...))
    end
end
println("vector-valued: ", npoints(vector_learner), " samples, first value ", result(vector_learner).values[1])
