# DataSaver: learn y while keeping the error estimate returned by the function.
using AdaptivePort

f(x) = (y = tanh(50x), err = 1e-3 * abs(x))
saver = DataSaver(Learner1D(f; bounds = (-1, 1)); arg_picker = r -> r.y)
runner = BlockingRunner(saver; max_npoints = 60)
extra = extra_data(saver)
println(length(extra), " results stored; err at x = ", first(keys(extra)), ": ", extra[first(keys(extra))].err)
