if [ -f ".env" ]; then
  source ".env"
fi

# Exit if environment variables are not set
if [ -z "$GEMINI_API" ]; then
  echo "Error: GEMINI_API is not set. Please check .env file."
  exit 1
fi

python python/1_train_dataset_RAG_judge_labeling.py \
    --api_key "$GEMINI_API" \
    --input_file "dataset/dataset_0_raw_train_dataset/0_raw_train_dataset.json" \
    --output_dir "dataset/dataset_1_train_dataset/1_train_dataset_constructed" \
    --concurrency 1 \
    --model_name "gemini-2.0-flash"
