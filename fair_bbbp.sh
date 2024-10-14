#!/bin/bash
#PBS -m abe
#PBS -M s5323328@griffithuni.edu.au
#PBS -N fair_comp_bb
#PBS -q gpuq2
#PBS -l select=1:ncpus=8:ngpus=1:mem=60gb,walltime=100:00:00

cd /export/home/s5323328/pakdd/MolCLR

source /usr/local/bin/s3proxy.sh
module load anaconda3/2024.06
source activate molclr
module load cuda/11.4


for i in {1..10}; do
    python finetune.py --dataset bbbp --model galactica-6.7b --knowledge_type all  --num_samples 30 --feat_type plus
    python finetune.py --dataset bbbp --model galactica-6.7b --knowledge_type all  --num_samples 30 --feat_type dir_concat
    python finetune.py --dataset bbbp --model galactica-6.7b --knowledge_type all  --num_samples 30 --feat_type concat
done

for i in {1..10}; do
    python finetune.py --dataset bbbp --model galactica-6.7b --knowledge_type all  --num_samples 50 --feat_type plus
    python finetune.py --dataset bbbp --model galactica-6.7b --knowledge_type all  --num_samples 50 --feat_type dir_concat
    python finetune.py --dataset bbbp --model galactica-6.7b --knowledge_type all  --num_samples 50 --feat_type concat
done