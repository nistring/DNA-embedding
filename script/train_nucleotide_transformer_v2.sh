#!/bin/bash

# ========================================
# Nucleotide Transformer v2 500m Distributed Training Script
# ========================================

# Configuration
export CUDA_VISIBLE_DEVICES=1  # Adjust based on available GPUs
export TORCH_DISTRIBUTED_BACKEND=nccl
export OMP_NUM_THREADS=1

# Number of GPUs to use
NUM_GPUS=1

# Model and data paths
MODEL_NAME="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
DATA_PATH="./data"  # Update with your data directory
RUN_NAME="nucleotide_transformer_v2_finetune_50m"
OUTPUT_DIR="./output/${RUN_NAME}"

# Training hyperparameters based on model specs
# NT-v2 500m: max_length=2048, trained with 1M token effective batch size
MAX_LENGTH=175
BATCH_SIZE=256  # Reduced for multi-GPU memory efficiency
GRAD_ACCUM=1   # Gradient accumulation to simulate larger effective batch
LEARNING_RATE=1e-4
EPOCHS=5
SEED=42

# Training options
# Based on pretraining: linear warmup over 16k steps, square root decay
WARMUP_STEPS=0    # Adjusted for downstream task
EVAL_STEPS=100      # Evaluation frequency
LOGGING_STEPS=50    # Logging frequency
EVAL_STRATEGY="epoch"  # Evaluate based on eval_steps
LR_SCHEDULER_TYPE="linear"  # Alternative: sqrt decay for better stability

echo "Starting ${RUN_NAME} distributed training with Nucleotide Transformer v2 500m..."
echo "Settings: LR=${LEARNING_RATE}, Batch=${BATCH_SIZE}, GradAccum=${GRAD_ACCUM}, MaxLen=${MAX_LENGTH}"
echo "Epochs=${EPOCHS}, Scheduler=${LR_SCHEDULER_TYPE}, Warmup=${WARMUP_STEPS}"
mkdir -p ${OUTPUT_DIR}

torchrun --nproc_per_node=${NUM_GPUS} train.py \
    --model_name_or_path ${MODEL_NAME} \
    --model_type "nucleotide-transformer-v2" \
    --use_reverse_complement True \
    --pcc_loss_alpha 1 \
    --bf16 --bf16_full_eval \
    --clinvar_csv ${DATA_PATH}/clinvar_compact_removed.csv \
    --clinvar_sep "," \
    --refs_csv ${DATA_PATH}/bac_refs.csv \
    --refs_sep "," \
    --run_name ${RUN_NAME} \
    --output_dir ${OUTPUT_DIR}/joint \
    --model_max_length ${MAX_LENGTH} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${EPOCHS} \
    --max_grad_norm 1.0 \
    --save_strategy ${EVAL_STRATEGY} \
    --eval_strategy ${EVAL_STRATEGY} \
    --eval_steps ${EVAL_STEPS} \
    --save_steps ${EVAL_STEPS} \
    --do_train True \
    --do_eval True \
    --logging_steps ${LOGGING_STEPS} \
    --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
    --warmup_steps ${WARMUP_STEPS} \
    --overwrite_output_dir True \
    --log_level info \
    --seed ${SEED} \
    --find_unused_parameters False \
    --dataloader_num_workers 48 \
    --dataloader_pin_memory True \
    --remove_unused_columns False \
    --weight_decay 0.01
    # > "${OUTPUT_DIR}/training.log" 2>&1 &

echo "Training started! Monitor progress at ${OUTPUT_DIR}/training.log"
echo "Model will be saved to ${OUTPUT_DIR}/joint"
