#!/bin/bash
# Submit from the repository root: sbatch examples/phase_diagram/job_mpi.sh
# Adjust partition, account and time to your cluster. MPI.jl must use the cluster's MPI library; once:
#   julia --project=examples -e 'using Pkg; Pkg.add("MPIPreferences"); using MPIPreferences; MPIPreferences.use_system_binary()'
#SBATCH --job-name=phase-diagram-mpi
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32         # 128 ranks: rank 0 runs the learner, 127 evaluate
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00

# run once on the login node first: julia --project=examples -e 'using Pkg; Pkg.instantiate()'
# precompile once here, not in 128 ranks at the same time
julia --project=examples -e 'using Pkg; Pkg.precompile()'

srun julia --project=examples examples/phase_diagram/run_mpi.jl
