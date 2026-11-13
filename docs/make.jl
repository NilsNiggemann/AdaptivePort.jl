using Documenter, AdaptivePort

makedocs(; sitename = "AdaptivePort.jl", modules = [AdaptivePort], pages = ["Home" => "index.md"])

deploydocs(; repo = "github.com/NilsNiggemann/AdaptivePort.jl.git", devbranch = "main")
