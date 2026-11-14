#!/bin/bash
# Submit from the repository root: sbatch examples/phase_diagram/job_distributed.sh
# Adjust partition, account and time to your cluster.
#SBATCH --job-name=phase-diagram
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32         # 4 x 32 = 128 Distributed workers
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00

# run once on the login node first: julia --project=examples -e 'using Pkg; Pkg.instantiate()'
# precompile once here, not in 128 workers at the same time
julia --project=examples -e 'using Pkg; Pkg.precompile()'

# the driver runs once; SlurmManager starts the workers on all nodes itself
julia --project=examples examples/phase_diagram/run_distributed.jl
