using Documenter, AdaptivePort

pages = [
    "Introduction" => ["Overview" => "index.md", "Getting started" => "getting_started.md"],
    "Examples" => ["Plotting results" => "examples/plotting.md", "Phase diagram on 4 nodes" => "examples/phase_diagram.md"],
    "Learners" => [
        "Learner1D" => "learners/learner1d.md",
        "Learner2D" => "learners/learner2d.md",
        "LearnerND" => "learners/learnernd.md",
        "AverageLearner" => "learners/average_learner.md",
        "AverageLearner1D" => "learners/average_learner1d.md",
        "IntegratorLearner" => "learners/integrator.md",
        "SequenceLearner" => "learners/sequence.md",
        "BalancingLearner" => "learners/balancing.md",
        "DataSaver" => "learners/datasaver.md",
        "Common interface" => "learners/interface.md",
    ],
    "Runners" => ["Running a learner" => "runners/runners.md", "Parallel execution" => "runners/parallel.md", "Clusters" => "runners/clusters.md"],
    "Saving data" => "saving.md",
    "Reference" => ["Compatibility with Python Adaptive" => "compatibility.md", "API reference" => "api.md"],
]
makedocs(; sitename = "AdaptivePort.jl", modules = [AdaptivePort], pages)

deploydocs(; repo = "github.com/NilsNiggemann/AdaptivePort.jl.git", devbranch = "main")
