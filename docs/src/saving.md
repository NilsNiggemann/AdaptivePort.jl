# Saving and export

Learners never store the function, so saving means writing the told data and replaying it later.

```julia
table = to_table(learner)            # named tuple of equal-length vectors, one row per tell!
save_data("run.csv", learner)        # or "run.jls"
fresh = Learner1D(f; bounds = (-1, 1))
load_data!(fresh, "run.csv")         # replays tell!; throws ArgumentError if the file is missing
```

`examples/save_load.jl` checkpoints a run, stops it on a time goal, reloads and continues. A runner can
write the file itself with `checkpoint = path` ([Runners](runners.md#Checkpoint-and-resume)).

## Tables

[`to_table`](@ref) returns a Tables.jl column table (upstream `to_dataframe`), so `DataFrame(to_table(l))` and
`CSV.write(path, to_table(l))` work without this package depending on either. It never changes the learner.

| Learner | Columns |
|:--------|:--------|
| `Learner1D` | `x`, `y` sorted by `x` (vector output: `y_1`..`y_M`) |
| `Learner2D` | `x`, `y`, `value` in tell order |
| `LearnerND` | `x`, `y`, `z` for $N \le 3$, else `x_1`..`x_N`, then `value`; tell order |
| `AverageLearner` | `seed`, `y` |
| `AverageLearner1D` | `seed`, `x`, `y`, one row per sample |
| `IntegratorLearner` | `x`, `y` |
| `SequenceLearner` | `index` (1-based), `x`, `y` |
| `BalancingLearner` | `learner_index` (1-based) plus the child columns, children stacked |
| `DataSaver` | the child's columns plus `extra_data` |

## CSV or `.jls`

The extension picks the format. `.csv` is a header and one number per cell, floats in shortest round-trip
form (nothing is lost), readable by pandas and NumPy. Anything else (use `.jls`) is a `Serialization` file
holding the table with a format tag and version; it is much faster (1e6 points: about 1 s as CSV, 0.09 s as
`.jls`, see [Performance](index.md#Performance)) and can hold non-numeric columns such as a
[`DataSaver`](@ref)'s `extra_data`.

Both are written to a temporary file and moved over the target, so a crash never leaves half a file.

## Limits

- Only told points are saved; pending points are not.
- `.jls` files depend on the Julia version, and loading one runs `deserialize`: load only your own files.
- `.csv` needs numeric columns; otherwise `save_data` throws `ArgumentError`.
- After `load_data!` the data are identical, but later sampling can differ from an uninterrupted run where a
  learner's state depends on history (the lazy $y$-scale of `Learner1D`), as upstream.
- `load_data!` needs a fresh learner built with the same function and bounds.
