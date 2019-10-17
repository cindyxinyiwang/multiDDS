#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=1000:00:00
#SBATCH --nodes=1
#SBATCH --mem=12g
#SBATCH --job-name="multlin"
##SBATCH --mail-user=gneubig@cs.cmu.edu
##SBATCH --mail-type=ALL
##SBATCH --requeue
#Specifies that the job will be requeued after a node failure.
#The default is that the job will not be requeued.
set -e
#export PYTHONPATH="$(pwd)"                                                       
#export CUDA_VISIBLE_DEVICES="0" 

version=single
mkdir -p checkpoints/"$version"
#for f in `ls job_scripts/"$version"/ | grep -v .sh$`; do
for f in `ls job_scripts/"$version"/`; do
  f1=`basename $f .sh`
  echo $f1
  if [[ ! -e checkpoints/"$version"/$f1.started ]]; then
    echo "running $f1"
    touch checkpoints/"$version"/$f1.started
    hostname
    nvidia-smi
    ./job_scripts/"$version"/$f $1
  else
    echo "already started $f1"
  fi
done

