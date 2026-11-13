# Runners: let AdaptivePort evaluate the function for you.
# Run with: julia -t 4 --project=. examples/runners.jl   (-t 4 only matters for part C)
using AdaptivePort

f(x) = sin(8x) + 0.2cos(23x)

# A. Blocking: returns once the loss is below `tol` or `max_npoints` is reached.
learner = Learner1D(f; bounds=(-1, 1))
runner = BlockingRunner(learner; tol=0.02, max_npoints=500)
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

# C. Threaded: keep `ntasks` evaluations in flight on the threads of this process; `f` must be thread-safe.
busy_f(x) = (t = time(); while time() - t < 0.02 end; f(x))  # 20 ms of CPU work
for ntasks in (1, Threads.nthreads())
    l = Learner1D(busy_f; bounds=(-1, 1))
    t = @elapsed BlockingRunner(l; max_npoints=80, ntasks)
    println("C: ntasks=$ntasks on $(Threads.nthreads()) threads: $(round(t; digits=2)) s for $(npoints(l)) points")
end

# D. Time goal: stop after a fixed wall-clock time, whatever the loss.
l = Learner1D(slow_f; bounds=(-1, 1))
BlockingRunner(l; duration=1.0)
println("D: $(npoints(l)) points in one second")
