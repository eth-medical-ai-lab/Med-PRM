#!/usr/bin/env python
# coding: utf-8
"""
Fine-tune LLM on PRM dataset (MedQA only) with optional RAG support.
"""

# =====================================================================
# 1. Import Libraries
# =====================================================================
import argparse
import torch
import json
import random
import os
import math
from datetime import datetime
import socket

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset

# --- Optional packages for wandb/huggingface_hub online features ----
try:
    import wandb
    from huggingface_hub import HfApi, create_repo
except ImportError:
    wandb = None
    HfApi = None

# =====================================================================
# 2. Argument Parser
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune LLM model on PRM dataset (MedQA only)."
    )

    # Model related
    parser.add_argument("--model_path", type=str, help="Path or name of the LLM model.")
    parser.add_argument("--device", type=str, help="CUDA visible devices (e.g. '0,1')")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type (default: bfloat16).")
    parser.add_argument("--max_token_len", type=int, default=1024, help="Maximum token length (default: 1023).")

    # Data related
    parser.add_argument("--train_json", type=str, help="Path to the training JSON file.")
    parser.add_argument("--train_ratio", type=float, default=1.0, help="Ratio for train/valid split (default: 1.0).")

    # Training hyperparameters
    parser.add_argument("--output_dir", type=str, help="Output directory.")
    parser.add_argument("--logging_steps", type=int, help="Logging steps.")
    parser.add_argument("--num_train_epochs", type=int, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, help="Learning rate.")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Gradient accumulation steps.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type.")
    parser.add_argument("--per_device_train_batch_size", type=int, help="Training batch size per device.")
    parser.add_argument("--bf16", type=bool, default=True, help="Whether to use bf16 training.")
    parser.add_argument("--run_name", type=str, help="Name for the run in logs.")
    parser.add_argument("--save_steps", type=int, help="Interval steps for saving checkpoints.")

    # Label types
    parser.add_argument("--train_label", type=str, 
                        choices=["prm_soft_label", "prm_hard_label", "gemini_label", "llama_label", "orm_label"],
                        help="Which training label to use.")

    # Filtering
    parser.add_argument("--do_filtering", type=str, default="yes",
                        help="Whether to perform filtering for orm_label=0/1. 'yes' or 'no'")

    # RAG usage (unique to 10_training_code.py)
    parser.add_argument("--use_rag", type=str, default="no",
                        choices=["yes", "no"],
                        help="Include related docs (RAG) if 'yes'.")

    # Online related
    parser.add_argument("--online", type=bool, default=False, help="Enable online mode (True/False)")
    parser.add_argument("--wandb_token", type=str, help="Wandb API token for online mode.")
    parser.add_argument("--wandb_project", type=str, help="Wandb project name for online mode.")
    parser.add_argument("--hf_token", type=str, help="HuggingFace Hub token for online mode.")

    return parser.parse_args()

# =====================================================================
# 3. Model & Tokenizer Loader
# =====================================================================
def load_model(model_name="meta-llama/Llama-3.1-8B-Instruct", dtype="bfloat16"):
    print("Loading model...")
    print(f"Model being loaded: {model_name}")  # Debug output added
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")  # Tokenizer info output
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    print(f"Model loaded: {model.__class__.__name__}, Parameters: {model.num_parameters():,}")  # Model info output
    model.gradient_checkpointing_enable()
    
    # Handle models without pad_token (e.g., LLaMA series)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")  # pad_token setting info
    
    print("Model loaded successfully.")
    return model, tokenizer

# =====================================================================
# 4. Utility Functions
# =====================================================================
def process_gemini_label(label_list):
    processed, found_zero = [], False
    for v in label_list:
        processed.append(0 if found_zero else v)
        if v == 0:  # All zeros after this
            found_zero = True
    return processed

# --- Additional: Truncate related docs within token budget ----------
def truncate_related_docs(docs, tokenizer,
                          max_total_len: int,
                          reserve_for_prompt: int = 1024):
    """
    docs: list[str] (raw docs)
    max_total_len: Total token limit
    reserve_for_prompt: Tokens to reserve for question/explanation/system prompt
    """
    kept, used = [], 0
    budget = max_total_len - reserve_for_prompt
    for d in docs:
        dtok = len(tokenizer(d, add_special_tokens=False)["input_ids"])
        if used + dtok + 1 > budget:
            break
        kept.append(d)
        used += dtok + 1
    return kept

def format_question_with_options(item):
    q = item.get("question", "")
    opts = item.get("options", [])
    if not opts:
        return q
    formatted = "".join(f" ({chr(ord('A')+i)}) {o}" for i, o in enumerate(opts))
    return q + formatted

# =====================================================================
# 5. JSON → Dataset Conversion
# =====================================================================
def process_all_results_to_dataset(all_results, tokenizer, step_tag_id=128256):

    data = {"input_ids": [], "attention_mask": [], "labels": [], "values": []}
    errors = 0

    for entry in all_results:
        raw = entry["query"].replace(" ки\n", " ки")  # Clean line breaks
        enc = tokenizer(raw, add_special_tokens=True, truncation=True)

        ids = enc["input_ids"] + [tokenizer.pad_token_id]
        attn = enc["attention_mask"] + [0]

        raw_label = raw.replace(" ки", " +")
        ref = tokenizer(raw_label, add_special_tokens=True)["input_ids"]
        ref.insert(0, tokenizer.pad_token_id)
        ref[0] = -100

        vals, cnt, ann = [], 0, entry["annotation"]
        if len(ref) != len(ids):
            errors += 1
            continue

        skip = False
        for j in range(len(ref)):
            if j == 0:
                vals.append(0)
                continue
            if ids[j - 1] == step_tag_id:
                if cnt < len(ann):
                    vals.append(ann[cnt]); cnt += 1
                else:
                    errors += 1; skip = True; break
            else:
                vals.append(0); ref[j] = -100
        if skip:  # Insufficient annotation length
            continue

        data["input_ids"].append(ids)
        data["attention_mask"].append(attn)
        data["labels"].append(ref)
        data["values"].append(vals)

    print(f"⚠️  Skipped due to errors: {errors} solution(s)")
    return data

def build_dataset_from_json(data, tokenizer, step_tag_id: int, args):
    """
    Convert MedQA samples to LLM input
    """
    RAG_SYSTEM_PROMPT = (
    "You are an evaluator assessing the logicality and validity of the reasoning in each step of the given explanation. "
    "In order to support the evaluation, the relevant documents, the question, and the explanation are provided sequentially. "
    "If the reasoning contains errors, output - after that step. If the reasoning in a step is logical and valid, output + after that step."
            )
    ORM_SYSTEM_PROMPT = (
        "You are an evaluator assessing the overall quality and correctness of the final answer in the given explanation. "
        "In order to support the evaluation, the question and the explanation are provided. "
        "If the final answer is incorrect or not well-supported, output -. If the final answer is correct and well-supported, output +."
            )
    PRM_SYSTEM_PROMPT = (
        "You are an evaluator assessing the logicality and validity of the reasoning in each step of the given explanation. "
        "In order to support the evaluation, the question and the explanation are provided. "
        "If the reasoning contains errors, output - after that step. If the reasoning in a step is logical and valid, output + after that step."
            )
    if args.use_rag == "yes":
        SYSTEM_PROMPT = RAG_SYSTEM_PROMPT
    elif args.train_label == "orm_label":
        SYSTEM_PROMPT = ORM_SYSTEM_PROMPT
    else:
        SYSTEM_PROMPT = PRM_SYSTEM_PROMPT

    all_results = []

    for idx, item in enumerate(data):
        qid = item.get("question_id", f"q_{idx}")
        q_formatted = format_question_with_options(item)
        sols = item.get("solutions", [])
        if not sols:
            continue

        # --- RAG document processing --------------------------------------
        if args.use_rag == "yes":
            docs_kept = truncate_related_docs(
                item.get("related_docs", []),
                tokenizer,
                max_total_len=args.max_token_len,
                reserve_for_prompt=1024,
            )
            doc_block = "".join(
                f"Document {i+1}: {d}\n\n" for i, d in enumerate(docs_kept)
            )
        else:
            doc_block = ""

        for sidx, sol in enumerate(sols):

            # Label extraction ------------------------------------------------
            if args.train_label == "gemini_label":
                arr = sol.get("prm_gemini_label", [])
                arr = arr if isinstance(arr, list) else [arr]
                label = process_gemini_label(arr)
            elif args.train_label == "llama_label":
                arr = sol.get("prm_llama_label", [])
                arr = arr if isinstance(arr, list) else [arr]
                label = process_gemini_label(arr)
            elif args.train_label == "orm_label":
                label = sol.get("orm_label", [])
                label = label if isinstance(label, list) else [label]
            else:
                label = sol.get(args.train_label, [])
                label = label if isinstance(label, list) else [label]

            # Solution text selection: ORM label case and others
            if args.train_label == "orm_label":
                sol_text = sol.get("orm_processed_solution", "")
            else:
                sol_text = sol.get("prm_processed_solution", "")

            # Input template --------------------------------------------
            content = (
                f"{doc_block}Question: {q_formatted}\n\nExplanation: {sol_text}"
            )
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
            raw_text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

            all_results.append(
                {
                    "query": raw_text,
                    "annotation": label,
                    "question_id": qid,
                    "solution_index": sidx,
                }
            )

    return process_all_results_to_dataset(
        all_results, tokenizer, step_tag_id=step_tag_id
    )

# =====================================================================
# 6. JSON Split (Maintain MedQA Filter)
# =====================================================================
def split_json_data(json_path, args):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Original code: MedQA only filtering
    # medqa = [d for d in data if d.get("data_source") == "med_qa"]
    # print(f"Total: {len(data)} / MedQA: {len(medqa)}")
    
    # Use all data
    print(f"Total data count: {len(data)}")

    # Perform filtering logic only when do_filtering == "yes"
    if args.do_filtering.lower() == "yes":
        for item in data:  # Use data instead of medqa
            if "solutions" not in item:
                continue
            
            solutions = item["solutions"]
            
            orm_0_solutions = []
            orm_1_solutions = []
            
            for sol in solutions:
                orm_label_val = sol.get("orm_label", 0)
                if orm_label_val == 0:
                    # Skip 0/1 check for er_label
                    if args.train_label == "er_label":
                        orm_0_solutions.append(sol)
                        continue
                    
                    # gemini/hard/soft/llama
                    if args.train_label == "gemini_label":
                        arr = sol.get("prm_gemini_label", [])
                        if not isinstance(arr, list):
                            arr = [arr]
                        arr = process_gemini_label(arr)
                    elif args.train_label == "llama_label":
                        arr = sol.get("prm_llama_label", [])
                        if not isinstance(arr, list):
                            arr = [arr]
                        arr = process_gemini_label(arr)
                    else:
                        # soft/hard
                        arr = sol.get(args.train_label, [])
                        if not isinstance(arr, list):
                            arr = [arr]
                    
                    if any(x == 0 for x in arr):
                        orm_0_solutions.append(sol)
                    else:
                        # Remove if all are 1
                        pass
                else:
                    orm_1_solutions.append(sol)
            
            remain_0_count = len(orm_0_solutions)
            need_1_count = max(remain_0_count, 2)
            
            keep_1_solutions = orm_1_solutions[:need_1_count]
            
            item["solutions"] = orm_0_solutions + keep_1_solutions

    # Use all data for training, ignoring train_ratio
    return data, []  # Return data instead of medqa

# =====================================================================
# 7. Custom Trainer (Same Loss)
# =====================================================================
class AutoRegressiveTrainer(Trainer):
    def __init__(self, my_tokenizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.my_tokenizer = my_tokenizer
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"]
        
        logits = logits[..., :-1, :].contiguous().to(torch.bfloat16)
        labels = labels[..., 1:].contiguous()
        values = inputs['values'][..., 1:].contiguous().to(torch.bfloat16)
        
        plus_id = self.my_tokenizer.encode(' +')[-1]
        minus_id = self.my_tokenizer.encode(' -')[-1]
        plus_logits = logits[:, :, plus_id]
        minus_logits = logits[:, :, minus_id]
        
        if labels.dim() == plus_logits.dim() - 1:
            labels = labels.unsqueeze(-1)
            values = values.unsqueeze(-1)
        
        chosen = (labels != -100)
        pred_plus_values = plus_logits[chosen]
        pred_minus_values = minus_logits[chosen]
        gt_values = values[chosen]
        
        pred_combined = torch.stack((pred_plus_values, pred_minus_values), dim=1)
        gt_negative = 1 - gt_values
        gt_combined = torch.stack((gt_values, gt_negative), dim=1)
        
        loss = torch.nn.functional.cross_entropy(
            pred_combined,
            gt_combined,
            reduction="mean"
        )
        loss = loss.to(torch.bfloat16)
        return (loss, outputs) if return_outputs else loss

# =====================================================================
# 8. Main
# =====================================================================
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # ---- Online mode check -----------------------------------------
    # Handle args.online if not bool type
    if isinstance(args.online, str):
        online = args.online.lower() == 'true'
    else:
        online = bool(args.online)
    
    print(f"🌐 Online mode: {online}")

    # ---- W&B Initialization (Online Mode) ------------------------------------
    if online:
        os.environ["WANDB_API_KEY"] = args.wandb_token or ""
        wandb.login(key=args.wandb_token, relogin=True)
        wandb.init(project=getattr(args, "wandb_project", "prm-finetune"),
                   name=args.run_name or "prm-finetune",
                   config=vars(args))

    # 1) Model Loading ---------------------------------------------------
    print(f"DEBUG: Starting model loading - Model path: {args.model_path}")  # Debug output added
    model, tokenizer = load_model(
        model_name=args.model_path or "meta-llama/Llama-3.1-8B-Instruct",  # Using args.model_path
        dtype=args.dtype
    )
    print(f"DEBUG: Model loading completed - Model type: {type(model).__name__}")  # Debug output added

    # 2) Additional Special Tokens -------------------------------------------
    special_tokens = {"additional_special_tokens": [" ки"]}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    step_tag = "ки"
    step_tag_id = tokenizer.encode(f" {step_tag}")[-1]
    print("Step tag id:", step_tag_id)

    # 3) Data Loading -------------------------------------------------
    train_raw, valid_raw = split_json_data(args.train_json, args=args)
    train_dict = build_dataset_from_json(train_raw, tokenizer, step_tag_id, args)
    train_ds = Dataset.from_dict(train_dict)

    # Skip validation if not used
    if len(valid_raw) == 0:
        valid_ds = None
    else:
        valid_dict = build_dataset_from_json(valid_raw, tokenizer, step_tag_id, args)
        valid_ds = Dataset.from_dict(valid_dict)

    # 4) Padding/Cutting ----------------------------------------------------
    def pad_fn(ex, max_len=args.max_token_len, pad_val=0):
        iid, attn, lab, val = ex["input_ids"], ex["attention_mask"], ex["labels"], ex["values"]
        if len(iid) < max_len:
            pad = max_len - len(iid)
            iid += [pad_val] * pad
            attn += [0] * pad
            lab += [-100] * pad
            val += [0] * pad
        else:
            iid, attn, lab, val = iid[:max_len], attn[:max_len], lab[:max_len], val[:max_len]
        return {
            "input_ids": iid,
            "attention_mask": attn,
            "labels": lab,  
            "values": val,
        }

    train_ds = train_ds.map(pad_fn)
    if valid_ds is not None:
        valid_ds = valid_ds.map(pad_fn)

    # 5) Trainer ----------------------------------------------------
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    run_name = (
        args.run_name
        or f"{os.path.basename(args.model_path)}_{args.train_label}_{args.learning_rate}"
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="no",  # --- Modified: MedQA only / no eval ----
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        save_strategy="epoch",
        logging_steps=args.logging_steps,
        save_total_limit=1,
        run_name=run_name,
        bf16=args.bf16,
        remove_unused_columns=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        report_to=["wandb"] if online else ["none"],
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=0.05,
    )

    trainer = AutoRegressiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,  # Skip evaluation if None
        data_collator=data_collator,
        my_tokenizer=tokenizer,
    )

    # 6) Training -------------------------------------------------------
    print(
        f"● Fine-tuning start | lr={args.learning_rate} | filter={args.do_filtering} "
        f"| RAG={args.use_rag} | μ={args.risk_param}"
    )
    trainer.train()
    print("✅ Fine-tuning finished")

    # 7) Saving -------------------------------------------------------
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    date_str = datetime.now().strftime("%Y%m%d")
    filter_str = "filter" if args.do_filtering == "yes" else "nofilter"
    rag_str = "rag" if args.use_rag == "yes" else "norag"
    save_tag = f"{date_str}-{os.path.basename(args.model_path)}-{args.train_label}-{filter_str}-{rag_str}-e{args.num_train_epochs}"
    print(f"🚀 Saved: {args.output_dir}/{save_tag}")

    # ---- 8) HF Hub Upload (Online Mode) -------------------------------
    if online:
        os.environ["HF_HUB_TOKEN"] = args.hf_token
        api = HfApi(token=args.hf_token)
        repo_name = f"{api.whoami()['name']}/{save_tag}"
        try:
            create_repo(repo_name, exist_ok=True, private=False)
            trainer.model.push_to_hub(repo_name)
            tokenizer.push_to_hub(repo_name)
            print(f"🚀 Pushed to HF Hub: https://huggingface.co/{repo_name}")
        except Exception as e:
            print(f"HF upload failed: {e}")

    if online and wandb.run is not None:
        wandb.finish()

    print("🎉 Task completed!")

if __name__ == "__main__":
    main()