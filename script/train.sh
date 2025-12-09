#!/bin/bash

# ========================================
# GPN Distributed Training Script
# ========================================

# Configuration
export CUDA_VISIBLE_DEVICES=0,1  # Adjust based on available GPUs
export TORCH_DISTRIBUTED_BACKEND=nccl
export OMP_NUM_THREADS=1

# Number of GPUs to use
NUM_GPUS=2

# Model and data paths
MODEL_NAME="songlab/gpn-brassicales"
TOKENIZER_NAME="gonzalobenegas/tokenizer-dna-mlm"
DATA_PATH="./data"  # Update with your data directory
RUN_NAME="0.8"
OUTPUT_DIR="./output/${RUN_NAME}"

# Training hyperparameters
MAX_LENGTH=1024
BATCH_SIZE=96
GRAD_ACCUM=1
LEARNING_RATE=1e-4
EPOCHS=3           # Standard for most tasks
SEED=42

# Training options
WARMUP_STEPS=0     # Official standard
EVAL_STEPS=1854 # 3708      # Official evaluation frequency
LOGGING_STEPS=50 # Keep high to reduce logging overhead
EVAL_STRATEGY="no"  # Evaluate based on eval_steps
LR_SCHEDULER_TYPE="cosine"  # Custom inverse_sqrt scheduler with rapid decay

echo "Starting ${RUN_NAME} distributed training..."
echo "Settings: LR=${LEARNING_RATE}, Batch=${BATCH_SIZE}, GradAccum=${GRAD_ACCUM}, Epochs=${EPOCHS}, Scheduler=${LR_SCHEDULER_TYPE}, Warmup=${WARMUP_STEPS}"
mkdir -p ${OUTPUT_DIR}
torchrun --nproc_per_node=${NUM_GPUS} train.py \
    --model_name_or_path ${MODEL_NAME} \
    --tokenizer_name ${TOKENIZER_NAME} \
    --weight_decay 0.01 \
    --model_type GPN \
    --clinvar_csv ${DATA_PATH}/clinvar_compact_removed.csv \
    --clinvar_sep "," \
    --refs_fasta ${DATA_PATH}/hg38.fa \
    --test_csv ${DATA_PATH}/test.csv \
    --run_name ${RUN_NAME} \
    --output_dir ${OUTPUT_DIR}/joint \
    --model_max_length ${MAX_LENGTH} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${EPOCHS} \
    --max_grad_norm 1.0 \
    --save_strategy "epoch" \
    --eval_strategy ${EVAL_STRATEGY} \
    --do_train True \
    --do_eval False \
    --logging_steps ${LOGGING_STEPS} \
    --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
    --overwrite_output_dir True \
    --log_level info \
    --seed ${SEED} \
    --ddp_find_unused_parameters False \
    --dataloader_num_workers 48 \
    --dataloader_pin_memory True \
    --remove_unused_columns False \
    --bf16 --bf16_full_eval \
    --cos_loss_margin 0.8 \
    > "${OUTPUT_DIR}/training.log" 2>&1 &