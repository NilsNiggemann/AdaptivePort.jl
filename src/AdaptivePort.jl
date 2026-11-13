"""
    AdaptivePort

Julia port of Python Adaptive. Currently provides [`Learner1D`](@ref): adaptive sampling of a scalar function on an interval through an `ask`/`tell!` interface.
"""
module AdaptivePort

export Learner1D, ask, tell!, tell_pending!, loss, done, result, npoints, default_loss

include("learner1D.jl")

end
