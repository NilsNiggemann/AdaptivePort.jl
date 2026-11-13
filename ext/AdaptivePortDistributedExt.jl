module AdaptivePortDistributedExt

import AdaptivePort: DistributedExecutor
import AdaptivePort
using Distributed: CachingPool, AbstractWorkerPool, RemoteException, remotecall, workers

DistributedExecutor() = DistributedExecutor(CachingPool(workers()))

# `remotecall` on a pool takes a free worker (waiting if all are busy) and returns it to the pool when the call
# finishes, so with `ntasks == length(pool)` every worker gets one point at a time.
(e::DistributedExecutor{<:AbstractWorkerPool})(f, x) = remotecall(f, e.pool, x...)

AdaptivePort.default_ntasks(e::DistributedExecutor{<:AbstractWorkerPool}) = length(e.pool)

# `fetch` of a failed remote call throws a `RemoteException`; its `captured` field holds the original exception
# and the backtrace from the worker.
AdaptivePort.captured_exception(e::RemoteException, bt) = e.captured

end
