# Checkpoint a run, stop it on a time goal, reload the data into a fresh learner and continue.
# Run with: julia --project=. examples/save_load.jl
using AdaptivePort

f(x) = (sleep(0.002); tanh(50x) + 0.1x)  # stands in for an expensive function
path = joinpath(mktempdir(), "run.csv")

# First session: three seconds of work (the first of them mostly compilation), checkpointed every 0.25 s and once more when the runner stops.
learner = Learner1D(f; bounds = (-1, 1))
BlockingRunner(learner; duration = 3, checkpoint = path, checkpoint_interval = 0.25)
println("first run: $(npoints(learner)) points, loss $(round(loss(learner); sigdigits = 3))")

# Second session: a new learner (the function is supplied again), resumed from the checkpoint.
resumed = Learner1D(f; bounds = (-1, 1))
isfile(path) && load_data!(resumed, path)
println("reloaded:  $(npoints(resumed)) points, identical data: $(result(resumed) == result(learner))")
BlockingRunner(resumed; tol = 0.005, checkpoint = path)
println("continued: $(npoints(resumed)) points, loss $(round(loss(resumed); sigdigits = 3))")

# The CSV is plain numbers: `pandas.read_csv(path)` or `DataFrame(CSV.File(path))` read it directly.
println(first(readlines(path), 3))
