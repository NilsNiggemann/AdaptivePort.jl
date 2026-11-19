# Clusters

On a cluster, the learner runs in one process and the evaluations are spread over many cores and nodes, with
either [Distributed.jl workers](parallel.md#Distributed.jl-workers) or [MPI ranks](parallel.md#MPI.jl). The
[phase diagram example](../examples/phase_diagram.md) is a complete setup for 4 nodes of a Slurm cluster with both;
it runs locally, and its job scripts are templates. This page collects the job-script patterns.

!!! tip "Checklist for a cluster run"
    1. Run `julia --project -e 'using Pkg; Pkg.instantiate()'` once on the login node (compute nodes often have
       no internet), and precompile in the job script before starting many processes:
       `julia --project -e 'using Pkg; Pkg.precompile()'`. Otherwise every process precompiles at the same time.
    2. Set `checkpoint = path`, so that a job killed at the time limit loses little.
    3. Set `duration` a few minutes below the job's time limit, so the runner stops cleanly and saves. With slow
       points, also set `drain` (the longest point time) and keep `duration + drain` below the limit: the runner
       then finishes the points in flight, and the workers are idle when the script ends or calls `rmprocs`.
       Removing workers that are still evaluating can crash them (seen as segfaults on Julia 1.12).
    4. Start with `isfile(path) && load_data!(learner, path)`, so that resubmitting the job continues the run.
    5. Check that each evaluation is slow enough to keep all workers busy
       (see [Number of workers](../examples/phase_diagram.md#Number-of-workers)).

## Distributed workers inside one allocation

[SlurmClusterManager.jl](https://github.com/JuliaParallel/SlurmClusterManager.jl) starts one worker per Slurm task,
on all nodes of the allocation. The driver script runs once:

```julia
# run.jl
using Distributed, SlurmClusterManager
addprocs(SlurmManager())                 # one worker per task of the allocation
@everywhere using AdaptivePort
@everywhere f(x) = expensive(x)          # f and its dependencies on every worker
learner = Learner1D(f; bounds = (-1, 1))
isfile("result.csv") && load_data!(learner, "result.csv")
BlockingRunner(learner; tol = 1e-3, duration = 110 * 60, executor = DistributedExecutor(), checkpoint = "result.csv")
```

```bash
#!/bin/bash
#SBATCH --job-name=adaptive
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
julia --project -e 'using Pkg; Pkg.precompile()'
julia --project run.jl                   # SlurmManager launches the workers itself
```

## MPI ranks

With [`MPIExecutor`](@ref) the script of the [MPI section](parallel.md#MPI.jl) is the job; Slurm starts all ranks
and rank 0 runs the learner:

```bash
#!/bin/bash
#SBATCH --job-name=adaptive-mpi
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --time=02:00:00
julia --project -e 'using Pkg; Pkg.precompile()'
srun --kill-on-bad-exit=1 julia --project mpi_script.jl       # or: mpiexec -n $SLURM_NTASKS julia --project mpi_script.jl
```

!!! warning "Use the cluster's MPI library"
    MPI.jl's bundled MPICH usually cannot use the cluster's network. Configure MPI.jl once, on the
    cluster, in the project you run (see the
    [MPI.jl docs](https://juliaparallel.org/MPI.jl/stable/configuration/)):

    ```bash
    julia --project -e 'using Pkg; Pkg.add("MPIPreferences"); using MPIPreferences; MPIPreferences.use_system_binary()'
    ```

All scripts on this page are templates; adjust partition, account and time to your cluster.
