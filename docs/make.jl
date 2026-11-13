using Documenter, AdaptivePort

pages = [
    "Home" => "index.md",
    "LearnerND" => "learnernd.md",
    "AverageLearner" => "average_learner.md",
    "AverageLearner1D" => "average_learner1d.md",
    "IntegratorLearner" => "integrator.md",
    "SequenceLearner" => "sequence.md",
]
makedocs(; sitename = "AdaptivePort.jl", modules = [AdaptivePort], pages)

deploydocs(; repo = "github.com/NilsNiggemann/AdaptivePort.jl.git", devbranch = "main")
