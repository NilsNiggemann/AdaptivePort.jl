using Documenter, AdaptivePort

pages = [
    "Home" => "index.md",
    "Learner1D" => "learner1d.md",
    "Learner2D" => "learner2d.md",
    "LearnerND" => "learnernd.md",
    "AverageLearner" => "average_learner.md",
    "AverageLearner1D" => "average_learner1d.md",
    "IntegratorLearner" => "integrator.md",
    "SequenceLearner" => "sequence.md",
    "Runners and parallel execution" => "runners.md",
    "Balancing" => "balancing.md",
    "DataSaver" => "datasaver.md",
    "Saving and export" => "saving.md",
    "Plotting" => "plotting.md",
    "Compatibility" => "compatibility.md",
    "API reference" => "api.md",
]
makedocs(; sitename = "AdaptivePort.jl", modules = [AdaptivePort], pages)

deploydocs(; repo = "github.com/NilsNiggemann/AdaptivePort.jl.git", devbranch = "main")
