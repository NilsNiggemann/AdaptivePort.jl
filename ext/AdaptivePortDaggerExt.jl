module AdaptivePortDaggerExt

using AdaptivePort, Dagger
import AdaptivePort: DaggerExecutor, captured_exception, default_ntasks

# Keywords go to `Dagger.Options` (`scope`, `proclist`, `occupancy`, ...) and apply to every evaluation.
DaggerExecutor(; options...) = DaggerExecutor(Dagger.Options(; options...))

(e::DaggerExecutor)(f, x) = Dagger.spawn(f, e.options, x...)

# Dagger's scheduler runs one task per processor (threads of every worker); respect the executor's scope option.
function default_ntasks(e::DaggerExecutor)
    scope = something(e.options.compute_scope, Dagger.DefaultScope())
    return Dagger.num_processors(scope)
end

# A failing task throws a DTaskFailedException (possibly nested) around a RemoteException around CapturedExceptions;
# report the original exception with the backtrace from inside `f`.
unwrap(e::Dagger.DTaskFailedException) = unwrap(e.ex)
unwrap(e::Dagger.Distributed.RemoteException) = unwrap(e.captured)
unwrap(e::CapturedException) = e.ex isa CapturedException || e.ex isa Dagger.Distributed.RemoteException ? unwrap(e.ex) : e
unwrap(e) = CapturedException(e, Any[])

function captured_exception(e::Dagger.DTaskFailedException, bt)
    c = unwrap(e)
    return isempty(c.processed_bt) ? CapturedException(c.ex, bt) : c
end

end
