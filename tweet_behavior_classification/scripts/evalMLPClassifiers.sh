#!/bin/sh
#$ -cwd
#$ -l mem_free=32g
#$ -l h_rt=72:00:00
#$ -l h_vmem=32g
#$ -l num_proc=8
#$ -N evalMLPClassifiers
#$ -S /bin/bash

DATA_SET=$1
CONNECT_INPUT_FLAG=$2

echo "Evaluating dataset ${DATA_SET} on all features"

if [ -z "${CONNECT_INPUT_FLAG}" ]
then
  echo "Call 'python nn_classification.py ${DATA_SET}'"
  python nn_classification.py ${DATA_SET}
else
  echo "python nn_classification.py --intoout ${DATA_SET}"
  python nn_classification.py --intoout ${DATA_SET}
fi
