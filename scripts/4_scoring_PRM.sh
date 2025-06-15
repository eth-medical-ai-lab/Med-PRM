#!/bin/bash

############################################
# Load Environment Variables
############################################
# Load environment variables from .env file
if [ -f ".env" ]; then
  source ".env"
fi

# Exit if environment variables are not set
if [ -z "$HF_TOKEN" ]; then
  echo "Error: HF_TOKEN is not set. Please check .env file."
  exit 1
fi

############################################
# User-defined Parameters
############################################
# Default settings

USE_RAG="yes"
USE_ORM="no"

# DATA_SOURCE_LIST='["med_qa"]' Process all if empty
PROCESS_SOLUTION_NUM=64

MODEL_PATHS=(
"model_train/llama-3.1-medprm-reward-v1.0"
)
INPUT_JSON="dataset/dataset_3_sampled_dataset/llama-3.1-medprm-reward-test-set/2_test_dataset.json"
# GPU number array for each model
GPUS=(0)

# Set output directory
OUTPUT_DIR="dataset/dataset_4_scored_dataset"

# Set maximum token length (1024 for other PRMs, 4096 for RAG-PRM)
MAX_TOKEN_LEN=4096

# Set option inclusion (yes/no)
INCLUDE_OPTIONS="no"

# Create log directory
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

BASE_INPUT_NAME="$(basename "$INPUT_JSON" .json)"
# Extract first data source element
FIRST_DATA_SOURCE=$(echo $DATA_SOURCE_LIST | sed -E 's/\[\"([^\"]+)\".*/\1/')

# Parallel execution for each model
for i in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH="${MODEL_PATHS[$i]}"
    GPU="${GPUS[$i]}"
    MODEL_BASENAME="${MODEL_PATH##*/}"
    OUTPUT_JSON="${OUTPUT_DIR}/${MODEL_BASENAME}_${FIRST_DATA_SOURCE}_sol${PROCESS_SOLUTION_NUM}_${BASE_INPUT_NAME}.json"
    LOG_FILE="${LOG_DIR}/TEST_$(date +'%Y%m%d_%H%M%S')_${MODEL_BASENAME}.log"
    
    echo "====== Evaluation Settings (Model: ${MODEL_BASENAME}, GPU: ${GPU}) ======" | tee -a "$LOG_FILE"
    echo "Model Name: $MODEL_NAME" | tee -a "$LOG_FILE"
    echo "Model Path: $MODEL_PATH" | tee -a "$LOG_FILE"
    echo "Output File: $OUTPUT_JSON" | tee -a "$LOG_FILE"
    echo "GPU: $GPU" | tee -a "$LOG_FILE"
    echo "Number of Solutions to Process: $SOLUTION_NUM" | tee -a "$LOG_FILE"
    echo "RAG Usage: $USE_RAG" | tee -a "$LOG_FILE"
    echo "Max Token Length: $MAX_TOKEN_LEN" | tee -a "$LOG_FILE"
    echo "Include Options: $INCLUDE_OPTIONS" | tee -a "$LOG_FILE"
    echo "ORM Usage: $USE_ORM" | tee -a "$LOG_FILE"
    echo "Data Source: $DATA_SOURCE_LIST" | tee -a "$LOG_FILE"
    echo "Number of Solutions to Process: $PROCESS_SOLUTION_NUM" | tee -a "$LOG_FILE"
    echo "====================" | tee -a "$LOG_FILE"
 
#        --data_source_list "$DATA_SOURCE_LIST" \
    # Pass single GPU number as --device argument to run on specified GPU
    python python/4_scoring_PRM.py \
        --model_save_path "$MODEL_PATH" \
        --input_json_file "$INPUT_JSON" \
        --output_json_file "$OUTPUT_JSON" \
        --device "$GPU" \
        --hf_token "$HF_TOKEN" \
        --use_rag "$USE_RAG" \
        --max_token_len "$MAX_TOKEN_LEN" \
        --include_options "$INCLUDE_OPTIONS" \
        --use_orm "$USE_ORM" \
        --process_solution_num "$PROCESS_SOLUTION_NUM" 2>&1 | tee -a "$LOG_FILE" &
done

wait
echo "All model evaluations completed."
