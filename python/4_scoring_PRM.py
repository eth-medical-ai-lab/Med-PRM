#!/usr/bin/env python
# coding: utf-8
"""
Run PRM evaluation with optional RAG support.
"""

import argparse
import os
import json
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import accelerate  # Retained from the original script.
from collections import Counter
# ----------------------------------------------------------------------
# 1. Argument parser
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PRM evaluation (RAG on/off selectable)."
    )

    # Model settings
    parser.add_argument("--model_save_path", type=str, required=True,
                        help="Path to the saved model directory")
    parser.add_argument("--device", type=str, default="",
                        help="CUDA visible devices (e.g. '0,1')")
    parser.add_argument("--hf_token", type=str, default="",
                        help="Hugging Face access token (optional)")

    # Data settings
    parser.add_argument("--input_json_file", type=str, required=True,
                        help="Path to input JSON file for evaluation")
    parser.add_argument("--output_json_file", type=str, required=True,
                        help="Path to save evaluation results")
    parser.add_argument("--process_solution_num", type=int, default=None,
                        help="Process only the first N solutions per question")
    parser.add_argument("--include_options", type=str,
                        choices=["yes", "no"], default="yes",
                        help="Include the options in the question text")

    # RAG settings
    parser.add_argument("--use_rag", type=str,
                        choices=["yes", "no"], default="yes",
                        help="'yes': use related_docs / 'no': base PRM only")
    parser.add_argument("--max_token_len", type=int, default=4096,
                        help="Token budget when use_rag is 'yes'")
    parser.add_argument("--use_orm", choices=["yes", "no"], default="no",
                   help="'yes': use orm_processed_solution when RAG is off") 
    parser.add_argument(
        "--data_source_list", type=str, default=None,
        help='JSON-array 형식으로 추론할 data_source 이름들만 지정 '
            '(예: \'["medqa","pubmedqa"]\'). 빈 리스트면 전체 사용'
    )       

    return parser.parse_args()


# ----------------------------------------------------------------------
# 2. Utility functions
# ----------------------------------------------------------------------
def format_question_with_options(item):
    """
    Build text in the form ``question (A) option 1 (B) option 2 ...``.
    """
    q = item.get("question", "")
    opts = item.get("options", [])
    if not opts:
        return q
    return q + "".join(f" ({chr(ord('A') + i)}) {opt}"
                       for i, opt in enumerate(opts))


def truncate_related_docs(docs, tokenizer,
                          max_total_len: int,
                          reserve_for_q_and_sol: int = 1024):
    """
    Truncate related documents to the token budget for RAG mode.
    """
    kept, used = [], 0
    budget = max_total_len - reserve_for_q_and_sol
    for doc in docs:
        tok_len = len(tokenizer(doc, add_special_tokens=False)["input_ids"])
        if used + tok_len + 1 > budget:
            break
        kept.append(doc)
        used += tok_len + 1
    return kept


# ----------------------------------------------------------------------
# 3. Main logic
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    raw_src_arg = args.data_source_list

    print("====== 평가 설정 ======")
    print(f"모델 경로: {args.model_save_path}")
    print(f"입력 파일: {args.input_json_file}")
    print(f"출력 파일: {args.output_json_file}")
    print(f"RAG 사용: {args.use_rag}")
    print(f"ORM 사용: {args.use_orm}")
    print("=====================")

    if not raw_src_arg:              # Use all sources for None or an empty string.
        filter_sources = []
    else:
        try:
            filter_sources = json.loads(raw_src_arg)
            assert isinstance(filter_sources, list)
        except Exception:
            raise ValueError("--data_source_list 는 JSON 배열 형식이어야 합니다 "
                            "(예: '[\"medqa\",\"mmlu\"]')")

    if args.hf_token:
        login(args.hf_token)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # Load the model and tokenizer.
    print("🔄 모델 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_save_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_save_path)
    print(f"✅ 모델 로드 완료: {type(model).__name__}")

    # Token IDs for '+' and '-'.
    plus_id = tokenizer(" +", add_special_tokens=False)["input_ids"][0]
    minus_id = tokenizer(" -", add_special_tokens=False)["input_ids"][0]
    print(f"plus_id  : {plus_id} ({tokenizer.convert_ids_to_tokens([plus_id])})")
    print(f"minus_id : {minus_id} ({tokenizer.convert_ids_to_tokens([minus_id])})")

    # --------------------------------------------------------------
    # Calculate PRM scores.
    # --------------------------------------------------------------
    def get_prob(text, special_char=" ки"):
        encoded = tokenizer(
            text, return_tensors="pt", return_offsets_mapping=True,
            add_special_tokens=True
        )
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        offsets = encoded["offset_mapping"][0]
        
        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask).logits[0]

        positions = [i for i, (s, e) in enumerate(offsets)
                     if text[s:e] == special_char]

        plus_probs, min_plus, final_plus = [], None, None
        for pos in positions:
            if pos >= logits.size(0):
                continue
            two = torch.stack([logits[pos][plus_id], logits[pos][minus_id]])
            probs = torch.softmax(two, dim=0)
            plus_probs.append(probs[0])
        if plus_probs:
            min_plus = torch.min(torch.stack(plus_probs)).item()
            final_plus = plus_probs[-1].item()
        
        return {
            "plus_probs": plus_probs,
            "min_plus_prob": min_plus,
            "final_plus_prob": final_plus
        }

    # --------------------------------------------------------------
    # Process JSON input.
    # --------------------------------------------------------------
    def process_json_with_prm():
        print("📂 JSON 파일 로드 중...")
        with open(args.input_json_file, encoding="utf-8") as f:
            data = json.load(f)
        
        if filter_sources:        # Ignore an empty filter list.
            data = [d for d in data if d.get("data_source") in filter_sources]
        total = len(data)
        print(f"📋 처리할 데이터 항목 수: {total}")

        # Select the system prompt for the active mode.
        RAG_SYSTEM_PROMPT = (
        "You are an evaluator assessing the logicality and validity of the reasoning in each step of the given explanation. "
        "In order to support the evaluation, the relevant documents, the question, and the explanation are provided sequentially. "
        "If the reasoning contains errors, output - after that step. If the reasoning in a step is logical and valid, output + after that step. "
                )

        PRM_SYSTEM_PROMPT = (
            "You are an evaluator assessing the logicality and validity of the reasoning in each step of the given explanation. "
            "In order to support the evaluation, the question and the explanation are provided. "
            "If the reasoning contains errors, output - after that step. If the reasoning in a step is logical and valid, output + after that step."
                )
        ORM_SYSTEM_PROMPT = (
            "You are an evaluator assessing the overall quality and correctness of the final answer in the given explanation. "
            "In order to support the evaluation, the question and the explanation are provided. "
            "If the final answer is incorrect or not well-supported, output -. If the final answer is correct and well-supported, output +."
                )
        # Track correctness immediately before the tqdm processing loop.
        prm_correct = 0          # Number correct using PRM
        mv_correct  = 0          # Number correct using majority voting
        
        with tqdm(total=total, desc="Processing Questions", unit="q") as pbar:
            for idx, item in enumerate(data):
                # Build the question text.
                q_text = (format_question_with_options(item)
                          if args.include_options == "yes"
                          else item.get("question", ""))

                # Limit the number of solutions.
                if args.process_solution_num is not None:
                    item["solutions"] = item["solutions"][:args.process_solution_num]
                sols = item["solutions"]

                # Preprocess documents in RAG mode.
                if args.use_rag == "yes":
                    docs = truncate_related_docs(
                        item.get("related_docs", []),
                        tokenizer,
                        max_total_len=args.max_token_len,
                        reserve_for_q_and_sol=1024
                    )
                    doc_block = "".join(f"Document {i+1}: {d}\n\n"
                                        for i, d in enumerate(docs))
                    system_prompt = RAG_SYSTEM_PROMPT
                    sol_key = "prm_processed_solution"
                else:  # RAG off
                    doc_block = ""
                    if args.use_orm == "yes":
                        system_prompt = ORM_SYSTEM_PROMPT
                        sol_key = "orm_processed_solution"
                    else:
                        system_prompt = PRM_SYSTEM_PROMPT
                        sol_key = "prm_processed_solution"

                # Assign PRM scores to each solution.
                for sol_idx, sol in enumerate(sols):
                    sol_text = sol.get(sol_key, "")
                    user_content = f"{doc_block}Question: {q_text}\n\nExplanation: {sol_text}"

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_content}
                    ]
                    raw = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    res = get_prob(raw, special_char=" ки")
                    plus_probs = [p.item() for p in res["plus_probs"]]
                    sol["PRM_min_score"] = res["min_plus_prob"] if res["min_plus_prob"] is not None else float("-inf")
                    sol["PRM_score"] = res["final_plus_prob"] if res["final_plus_prob"] is not None else float("-inf")
                    sol["PRM_score_list"] = plus_probs

                # ───────────────────────────────────────────
                # 2-1. Determine correctness using PRM.
                # ───────────────────────────────────────────
                valid = [s for s in sols if s["PRM_min_score"] != float("-inf")]
                prm_pred = max(valid, key=lambda s: s["PRM_min_score"]) if valid else None
                if prm_pred and prm_pred.get("score", 0) == 1:   # score == 1 means correct.
                    prm_correct += 1

                # ───────────────────────────────────────────
                # 2-2. Determine correctness using majority voting.
                #      - Select the most frequent answer string.
                #      - Mark it correct if any matching solution has score == 1.
                # ───────────────────────────────────────────
                
                if sols:                                          # Only when solutions are available.
                    most_common_ans, _ = Counter(s["answer"] for s in sols).most_common(1)[0]
                    mv_sols = [s for s in sols if s["answer"] == most_common_ans]
                    if any(s.get("score", 0) == 1 for s in mv_sols):
                        mv_correct += 1

                # Calculate and display accuracy after each question.
                current_prm_acc = (prm_correct / (idx + 1)) * 100
                current_mv_acc = (mv_correct / (idx + 1)) * 100
                
                
                # Display detailed progress information.
                pbar.set_description(f"Q{idx+1}/{total}")
                pbar.set_postfix(
                    PRM=f"{prm_correct}/{idx+1} ({current_prm_acc:.1f}%)",
                    MV=f"{mv_correct}/{idx+1} ({current_mv_acc:.1f}%)"
                )
                pbar.update(1)

        print("\n💾 결과 저장 중...")
        with open(args.output_json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"\n✅ Done. Results saved to {args.output_json_file}")
        print(f"PRM Accuracy : {prm_correct}/{total} ({100*prm_correct/total:.2f}%)")
        print(f"Maj-Vote Acc : {mv_correct}/{total} ({100*mv_correct/total:.2f}%)")
    # Run evaluation.
    process_json_with_prm()

if __name__ == "__main__":
    main()
