#!/bin/bash
if [ -f "../.env" ]; then
  source "../.env"
fi

# Exit if environment variables are not set
if [ -z "$HF_TOKEN" ]; then
  echo "Error: HF_TOKEN is not set. Please check .env file."
  exit 1
fi

MODEL_PATH="./model_downloaded/llama_3.1_8b_instruct"

# Input file specification
INPUT_FILE="./dataset/dataset_2_raw_test_dataset/0527_final_raw_test_dataset.json"

# Output directory for results
OUTPUT_DIR="./dataset/dataset_3_sampled_dataset"

# Specify data sources to process (comma-separated) / Process all if empty
# DATA_SOURCE_LIST=""

GPU_ID=0

REPEAT_COUNT=64
TEMPERATURE=0.7
TOP_K=50
TOP_P=0.9
MAX_TOKENS=4096

# Extract filename
FILENAME=$(basename "$INPUT_FILE")
# Extract model name (last part of path)
MODEL_NAME=$(basename "$MODEL_PATH")

# Create log directory
log_dir="../logs"
mkdir -p $log_dir

#   --data_source_list "$DATA_SOURCE_LIST" \
# Process single file on GPU 7 (normal execution)
python ../python/3_test_dataset_sampling.py \
  --hf_token "$HF_TOKEN" \
  --model_path "$MODEL_PATH" \
  --repeat_count "$REPEAT_COUNT" \
  --temperature "$TEMPERATURE" \
  --top_k "$TOP_K" \
  --top_p "$TOP_P" \
  --max_tokens "$MAX_TOKENS" \
  --input_file "$INPUT_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --gpu_id "$GPU_ID" \
  > "${log_dir}/test_${FILENAME}_${MODEL_NAME}_${REPEAT_COUNT}.log" 2>&1
