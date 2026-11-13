# Learn a noisy step-like function with AverageLearner1D: new x where the mean bends, more samples where the noise hurts.
# Run with: julia --project=. examples/average_learner1d.jl
using AdaptivePort, Random

# One noisy sample; the seed is the only source of randomness. The noise is larger on the right.
noisy(seed, x) = tanh(10x) + (0.05 + 0.2 * (x > 0)) * randn(Random.Xoshiro(hash((seed, x))))

learner = AverageLearner1D(noisy; bounds = (-1, 1), min_samples = 10)
BlockingRunner(learner; max_npoints = 40)  # max_npoints counts distinct x, not samples

r = result(learner)
println("distinct x: ", npoints(learner), ", samples: ", nsamples(learner), ", loss: ", round(loss(learner); digits = 4))
println("samples on the left (sigma 0.05):   ", sum(r.nsamples[r.x .< 0]))
println("samples on the right (sigma 0.25): ", sum(r.nsamples[r.x .> 0]))
println("largest mean error: ", round(maximum(filter(isfinite, r.error)); digits = 4))
