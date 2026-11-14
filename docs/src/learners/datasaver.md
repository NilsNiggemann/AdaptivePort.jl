# DataSaver

Some functions return more than the number to learn: an error estimate, a diagnostic, a whole simulation
record. [`DataSaver`](@ref) wraps a learner, teaches it `arg_picker(result)` and keeps the full `result` of
every point, retrievable with [`extra_data`](@ref).

```julia
using AdaptivePort

f(x) = (y = tanh(50x), err = 1e-3 * abs(x))
saver = DataSaver(Learner1D(f; bounds = (-1, 1)); arg_picker = r -> r.y)
BlockingRunner(saver; max_npoints = 60)
extra_data(saver)    # Dict: point => (y = ..., err = ...)
```

This is `examples/datasaver.jl`. Everything else (`ask`, `loss`, `result`, `npoints`, ...) is forwarded to
`saver.learner`, and the runner calls the child's function and tells the full result.

- Telling a repeated point overwrites the stored result, as upstream, even where the child keeps its first value.

!!! warning "Save to `.jls`, not `.csv`"
    `to_table(saver)` adds an `extra_data` column, which is not numeric, so [`save_data`](@ref) needs a `.jls`
    path. Saving `saver.learner` to CSV works but loses the extra results. See [Saving data](../saving.md).
