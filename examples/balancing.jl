# Balance several functions of different difficulty with one runner.
# Run with: julia --project=. examples/balancing.jl
using AdaptivePort

fs = [x -> x, x -> sin(3x), x -> tanh(50x), x -> sin(40x) * exp(-x^2)]
balancer = BalancingLearner([Learner1D(f; bounds = (-1, 1)) for f in fs])

BlockingRunner(balancer; max_npoints = 400, ntasks = 4)

for (i, l) in enumerate(balancer.learners)
    println("function ", i, ": ", npoints(l), " points, loss ", round(loss(l); sigdigits = 3))
end

# One child per parameter combination
steepness = AdaptivePort.from_product((x; k) -> tanh(k * x), Learner1D, (bounds = (-1, 1),), (k = [1, 10, 100],))
BlockingRunner(steepness; max_npoints = 150)
for (cd, l) in zip(steepness.cdims, steepness.learners)
    println(cd, ": ", npoints(l), " points")
end
