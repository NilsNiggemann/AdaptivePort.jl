# Evaluate a function on a fixed list of inputs with SequenceLearner.
# Run with: julia --project=. examples/sequence.jl
using AdaptivePort

inputs = ["a", "bb", "ccc", "dddd"]
learner = SequenceLearner(length; sequence = inputs)
BlockingRunner(learner; batch_size = 2)

data = result(learner)
for (i, y) in zip(data.indices, data.values)
    println(inputs[i], " -> ", y)
end
