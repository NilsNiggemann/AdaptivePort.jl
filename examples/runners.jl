# Runners: let AdaptivePort evaluate the function for you.
# Run with: julia -t 4 --project=. examples/runners.jl   (-t 4 only matters for part C)
using AdaptivePort

f(x) = sin(8x) + 0.2cos(23x)

# A. Blocking: returns once the loss is below `tol` or `max_npoints` is reached.
learner = Learner1D(f; bounds=(-1, 1))
runner = BlockingRunner(learner; tol=0.02, max_npoints=500, batch_size=1)
println("A: ", status(runner), " with ", npoints(learner), " points, loss ", loss(learner))

# B. Asynchronous: the runner works whenever this task yields (sleep, wait, I/O).
slow_f(x) = (sleep(0.01); f(x))
learner = Learner1D(slow_f; bounds=(-1, 1))
runner = AsyncRunner(learner; tol=0.02)
for _ in 1:3
    sleep(0.5)  # stands in for other work
    println("B: ", status(runner), ", ", npoints(learner), " points so far")
end
wait(runner)
println("B: ", status(runner), " with ", npoints(learner), " points")

# C. Parallel: each batch is evaluated on `Threads.nthreads()` threads; `f` must be thread-safe.
busy_f(x) = (t = time(); while time() - t < 0.02 end; f(x))  # 20 ms of CPU work
for parallel in (false, true)
    l = Learner1D(busy_f; bounds=(-1, 1))
    t = @elapsed BlockingRunner(l; max_npoints=80, batch_size=Threads.nthreads(), parallel)
    println("C: parallel=$parallel on $(Threads.nthreads()) threads: $(round(t; digits=2)) s for $(npoints(l)) points")
end
