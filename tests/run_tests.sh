#!/bin/bash
#PBS -N  SW_Test_Suite
##PBS -t 1-10
#PBS -M schwetli@uni-potsdam.de
#PBS -m ae
#PBS -j oe
#PBS -l ncpus=1
#PBS -l nodes=1:ppn=1
#PBS -l walltime=96:00:00
#PBS -l mem=2g

export NUM_CORES=$(cat $PBS_NODEFILE | wc -w | xargs)

export OMP_NUM_THREADS=$NUM_CORES

echo "Hi, I am $PBS_JOBID [$PBS_ARRAYID] $(hostname)."
echo "I am running on $NUM_CORES cores."

pwd

cd ..

pwd

python3 -m pytest test_sw_obj.py -m basictest

echo "I'm done now"
