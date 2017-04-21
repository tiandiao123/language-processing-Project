#!/bin/sh
#$ -cwd
#$ -l mem_free=4g
#$ -l h_rt=72:00:00
#$ -l h_vmem=4g
#$ -l num_proc=8
#$ -t 1-3
#$ -N evalBatchClassifiers
#$ -S /bin/bash

echo "Evaluating dataset #${SGE_TASK_ID}"

python -c "
import batch_classification as bc
import shared

dIdx = ${SGE_TASK_ID}-1
dName, dPath = shared.DSET_NAMES[dIdx], shared.FEATURE_PATHS[dIdx]
bc.evalAllFsets(dName, dPath)
"
