#!/usr/bin/env bash
# run_all.sh: executes all baseline and proposed scripts sequentially
# Usage: ./run_all.sh <dataset_dir> <class_name>

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <dataset_dir> <class_name>"
  exit 1
fi

DATASET_DIR=$1
CLASS_NAME=$2
GPU=${GPU:-0}
SEED=${SEED:-42}
MEMORY_SIZE=${MEMORY_SIZE:-10000}
NN=${ANOMALY_SCORER_NUM_NN:-1}
FAISS_WORKERS=${FAISS_NUM_WORKERS:-8}
SAMPLER_PARAMS="-p 0.1 approx_greedy_coreset"

# Baseline runs
echo "Running main_btf_raw"
python3 main_btf_raw.py

echo "Running main_btf_fpfh"
python3 main_btf_fpfh.py

# M3DM runs
echo "Running main_m3dm with Point_MAE"
python3 main_m3dm.py --xyz_backbone_name Point_MAE \
    --save_checkpoint_path ./checkpoints/pointmae_pretrain.pth

echo "Running main_m3dm with Point_Bert"
python3 main_m3dm.py --xyz_backbone_name Point_Bert \
    --save_checkpoint_path ./checkpoints/pointmae_pretrain.pth

# PatchCore runs
echo "Running main_patchcore_raw"
python3 main_patchcore_raw.py --gpu $GPU --seed $SEED \
    --memory_size $MEMORY_SIZE --anomaly_scorer_num_nn $NN \
    --faiss_on_gpu --faiss_num_workers $FAISS_WORKERS sampler $SAMPLER_PARAMS

echo "Running main_patchcore_fpfh_raw"
python3 main_patchcore_fpfh_raw.py --gpu $GPU --seed $SEED \
    --memory_size $MEMORY_SIZE --anomaly_scorer_num_nn $NN \
    --faiss_on_gpu --faiss_num_workers $FAISS_WORKERS sampler $SAMPLER_PARAMS

echo "Running main_patchcore_pointmae"
python3 main_patchcore_pointmae.py --gpu $GPU --seed $SEED \
    --memory_size $MEMORY_SIZE --anomaly_scorer_num_nn $NN \
    --faiss_on_gpu --faiss_num_workers $FAISS_WORKERS sampler $SAMPLER_PARAMS

# Regression-based 3DAD run
echo "Running main_reg3dad"
python3 main_reg3dad.py --gpu $GPU --seed $SEED \
    --memory_size $MEMORY_SIZE --anomaly_scorer_num_nn $NN \
    --faiss_on_gpu --faiss_num_workers $FAISS_WORKERS sampler $SAMPLER_PARAMS

# Proposed method runs
echo "Running proposed on Real3D-AD"
python3 main_proposed.py --dataset real3d --dataset_dir $DATASET_DIR \
    --class_name $CLASS_NAME --num_points 2048 --batch_size 8 --epochs 100 --lr 1e-4 --device cuda

echo "Running proposed on Industrial3D-AD"
python3 main_proposed.py --dataset industrial3d --dataset_dir $DATASET_DIR \
    --class_name $CLASS_NAME --num_points 2048 --batch_size 8 --epochs 100 --lr 1e-4 --device cuda

echo "All jobs completed."
