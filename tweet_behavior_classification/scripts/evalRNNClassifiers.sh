#!/bin/sh
#$ -cwd
#$ -l mem_free=32g
#$ -l h_rt=24:00:00
#$ -l h_vmem=32g
#$ -l num_proc=8
#$ -N evalRNNClassifiers
#$ -S /bin/bash

DATA_SET=$1

echo "Evaluating dataset ${DATA_SET} on all features"

python lstm_classification.py ${DATA_SET}
