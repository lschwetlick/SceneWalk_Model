#!/bin/bash
#SBATCH --comment="ext_corp_omp_log"
#SBATCH --mem=2G
#SBATCH --chdir=/mnt/scratch/sfb/1294/schwetli/scenewalk/PROJECTS/fitting_parameters/79_omp_corpus_log
#SBATCH --error="/mnt/scratch/sfb/1294/schwetli/o2/est_slurm-%j.%N.err"
#SBATCH --output="/mnt/scratch/sfb/1294/schwetli/o2/est_slurm-%j.%N.out"
#SBATCH --partition=Spc
#SBATCH --cpus-per-task=18
#SBATCH --array=1-35
#SBATCH --account=spc

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load Miscellaneous/1.0.1
module load python/3.7.6

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "I am ${SLURM_ARRAY_TASK_ID}"

pwd

source ../../../venv/sw_env0220_5/bin/activate
which python
python -c 'import sys; print(sys.path)'
python run_dream.py $SLURM_ARRAY_TASK_ID

echo "I'm done now"