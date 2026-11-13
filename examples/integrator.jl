# Adaptive quadrature with IntegratorLearner. Run: julia --project=. examples/integrator.jl
using AdaptivePort

# Integral of a function with a cusp-like singularity at 0: the exact value is 2.
f(x) = x^-0.5
learner = IntegratorLearner(f; bounds = (0, 1), tol = 1e-6)

# A runner stops by itself once the learner is done; no `tol` is needed.
BlockingRunner(learner)
res = result(learner)
println("integral = $(res.integral), error estimate = $(res.error), points = $(res.npoints)")
println("actual error = $(abs(res.integral - 2))")

# The manual loop.
learner = IntegratorLearner(x -> sin(20x)^2; bounds = (0, 3), tol = 1e-8)
while !AdaptivePort.done(learner)
    xs = ask(learner, 10)
    for x in xs
        tell!(learner, x, sin(20x)^2)
    end
end
res = result(learner)
exact = 3 / 2 - sin(120) / 80
println("integral = $(res.integral), error estimate = $(res.error), actual error = $(abs(res.integral - exact)), points = $(res.npoints)")
