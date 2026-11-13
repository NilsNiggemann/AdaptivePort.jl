# Learn the mean of a noisy function with AverageLearner; the runner stops once the mean is within atol.
# Run with: julia --project=. examples/average_learner.jl
using AdaptivePort, Random

noisy(seed) = 1.5 + randn(Random.Xoshiro(seed))  # the seed is the only source of randomness

learner = AverageLearner(noisy; atol = 0.05)
BlockingRunner(learner)

r = result(learner)
println("samples: ", r.n)
println("mean: ", round(r.mean; digits = 4), " +- ", round(r.std / sqrt(r.n); digits = 4), " (true 1.5)")
println("final loss: ", loss(learner))
