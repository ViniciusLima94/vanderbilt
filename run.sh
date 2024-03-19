#!/bin/bash

#SBATCH -J proc              # Job name
#SBATCH -o .out/job_%a.out   # Name of stdout output file (%j expands to %jobID)
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=40
#SBATCH --array=0-18


if [[ $1 == "savedata" ]]
then
    python -O SaveDownsampledData.py $2 $SLURM_ARRAY_TASK_ID
fi

if [[ $1 == "saveimfs" ]]
then
    python -O SaveIMFs.py $2 $SLURM_ARRAY_TASK_ID $3 0
fi

if [[ $1 == "gencomposites" ]]
then
    python -O GenerateCompositeSignals.py $2 $SLURM_ARRAY_TASK_ID $3 0
fi

if [[ $1 == "detectbursts" ]]
then
    python -O BurstDetection.py $2 $SLURM_ARRAY_TASK_ID $3 0
fi

if [[ $1 == "detectbursts3D" ]]
then
    python -O BurstDetection3D.py $2 $SLURM_ARRAY_TASK_ID $3 0
fi
