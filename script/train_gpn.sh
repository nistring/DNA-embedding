#!/bin/bash

# ========================================
# DNABERT-2 Distributed Training Script
# Official settings from DNABERT-2 paper and repository
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
RUN_NAME="gpn_finetune1"
OUTPUT_DIR="./output/${RUN_NAME}"

# Training hyperparameters
MAX_LENGTH=1024
BATCH_SIZE=128
GRAD_ACCUM=2
LEARNING_RATE=1e-3 # Official DNABERT-2 learning rate
EPOCHS=10           # Standard for most tasks
SEED=42

# Training options (official DNABERT-2 settings)
WARMUP_STEPS=10     # Official standard
EVAL_STEPS=50      # Official evaluation frequency
LOGGING_STEPS=10 # Keep high to reduce logging overhead
EVAL_STRATEGY="epoch"  # Evaluate based on eval_steps
LR_SCHEDULER_TYPE="linear"  # Custom inverse_sqrt scheduler with rapid decay

echo "Starting ${RUN_NAME} distributed training with joint ClinVar + mutation datasets..."
echo "Settings: LR=${LEARNING_RATE}, Batch=${BATCH_SIZE}, GradAccum=${GRAD_ACCUM}, Epochs=${EPOCHS}, Scheduler=${LR_SCHEDULER_TYPE}, Warmup=${WARMUP_STEPS}"
torchrun --nproc_per_node=${NUM_GPUS} train_dnabert2_ddp.py \
    --model_name_or_path ${MODEL_NAME} \
    --tokenizer_name ${TOKENIZER_NAME} \
    --soft_masked_loss_weight_train 0.1 --soft_masked_loss_weight_evaluation 0.0 \
    --weight_decay 0.01 \
    --model_type GPN \
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
    --do_train True \
    --do_eval True \
    --logging_steps ${LOGGING_STEPS} \
    --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
    --overwrite_output_dir True \
    --log_level info \
    --seed ${SEED} \
    --find_unused_parameters False \
    --dataloader_num_workers 1 \
    --dataloader_pin_memory True \
    --remove_unused_columns False \
    # --warmup_steps ${WARMUP_STEPS} \
    # --save_steps ${EVAL_STEPS} \
    # --eval_steps ${EVAL_STEPS}

echo "Training complete! Model saved to ${OUTPUT_DIR}/joint"