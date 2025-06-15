#!/usr/bin/env bash
##############################################################################
# Load Environment Variables
##############################################################################
# Load environment variables from .env file
if [ -f ".env" ]; then
  source ".env"
elif [ -f "../.env" ]; then
  source "../.env"
fi

# Exit if environment variables are not set
if [ -z "$HF_TOKEN" ]; then
  echo "Error: HF_TOKEN is not set. Please check .env file."
  exit 1
fi

if [ -z "$WANDB_TOKEN" ]; then
  echo "Error: WANDB_TOKEN is not set. Please check .env file."
  exit 1
fi


# Main settings
use_rag="yes"
train_label="gemini_label"

max_token_len=4096

num_train_epochs=3
do_filtering="yes"

# Environment-specific settings
gpu="0"
online=True


##############################################################################
# Basic Configuration Variables
##############################################################################
# Hugging Face and W&B token settings
hf_token="$HF_TOKEN"
wandb_token="$WANDB_TOKEN"
wandb_project="Med-PRM"

# Model path settings
model_name="meta-llama/Llama-3.1-8B-Instruct"
# Path settings
dataset_path="dataset/dataset_1_train_dataset/llama-3.1-medprm-reward-training-set/1_train_dataset.json"
base_output_dir="model_train"

# Check if dataset file exists
if [ ! -f "$dataset_path" ]; then
  echo "Error: Dataset file ($dataset_path) does not exist."
  exit 1
fi

# Check and create output directory if it doesn't exist
if [ ! -d "$base_output_dir" ]; then
  echo "Output directory ($base_output_dir) does not exist. Creating it now."
  mkdir -p "$base_output_dir"
fi

# Training parameter settings
lr_scheduler_type="cosine"
per_device_train_batch_size=1
gradient_accumulation_steps=64
bf16=True
timestamp=$(date +%Y%m%d_%H%M%S)
logging_steps=1
save_steps=50000
dtype="bfloat16"
train_ratio=1.0
# Training label and hyperparameter settings

learning_rate=2e-6
risk_param=5.0


# Create log directory
log_dir="logs"
mkdir -p $log_dir

##############################################################################
# Training Execution
##############################################################################
# Extract model name


# Set model save path
output_dir="${base_output_dir}/${model_name}-${train_label}-filter_${do_filtering}-ep${num_train_epochs}-${timestamp}-RAG_${use_rag}"
run_name="finetune_${model_name}_${train_label}_filter_${do_filtering}_ep${num_train_epochs}_${timestamp}-RAG_${use_rag}"
log_file="${log_dir}/TRAIN_${train_label}_${learning_rate}_${timestamp}-RAG_${use_rag}.log"

# Check Python script path
python_script="python/2_train.py"

# Log start message
echo "1: Training started" > "$log_file"


# Execute training on single GPU (redirect output to log file)
TRANSFORMERS_NO_DEEPSPEED=1 \
CUDA_VISIBLE_DEVICES=${gpu} python "$python_script" \
    --model_path "$model_name" \
    --device "$gpu" \
    --dtype "$dtype" \
    --max_token_len "$max_token_len" \
    --train_json "$dataset_path" \
    --train_ratio "$train_ratio" \
    --output_dir "$output_dir" \
    --logging_steps $logging_steps \
    --num_train_epochs $num_train_epochs \
    --learning_rate $learning_rate \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --lr_scheduler_type "$lr_scheduler_type" \
    --per_device_train_batch_size $per_device_train_batch_size \
    --bf16 $bf16 \
    --run_name "$run_name" \
    --save_steps $save_steps \
    --train_label "$train_label" \
    --risk_param "$risk_param" \
    --do_filtering "$do_filtering" \
    --use_rag "$use_rag" \
    --online $online \
    --wandb_token "$wandb_token" \
    --wandb_project "$wandb_project" \
    --hf_token "$hf_token" \
    2>&1 | tee -a "$log_file"

echo "Training completed. Logs are saved in ${log_file}."
