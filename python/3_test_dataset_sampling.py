#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vLLM-based multi-question generation + PRM/ORM 후처리
 - 선택형 데이터(예: med_qa, medmc_qa …)와
   주관식 데이터(예: nejm, osce) 를 한 파이프라인에서 처리
 - 최종 JSON 구조 예시는 README 하단 참조
"""
import argparse
import os
import json
import re
import math
import string
import torch
from collections import defaultdict

from huggingface_hub import login
from vllm import LLM, SamplingParams

# ──────────────────────────────────────────────────────────────
# 0. 상수
# ──────────────────────────────────────────────────────────────
MULTIPLE_CHOICE_SOURCES = {"med_qa", "medmc_qa", "ddxplus", "mmlu_anatomy", 
                          "mmlu_clinical_knowledge", "mmlu_college_biology", 
                          "mmlu_college_medicine", "mmlu_medical_genetics", 
                          "mmlu_professional_medicine", "pubmed_qa"}            
OPEN_SOURCES           = {"nejm", "osce"}                   # 주관식

SYSTEM_PROMPT = (
    "Solve the following question step-by-step. "
    "Do not analyze individual options in a single step. "
    "Each step of your explanation must start with 'Step {number}:' format. "
    "You must provide the answer using the phrase 'the answer is (option alphabet)' at the end of your step."
)

OPEN_SYSTEM_PROMPT = (
"Solve the following question step-by-step. Each step of your explanation must start with \'## Step {number}: \' format. The final answer must output a concise and clearly defined diagnostic term.  You must provide the final answer using the phrase \'## Final Diagnosis: {Disease name}\' at the end of your final step. Please refer to the following example. ## Final Diagnosis: Multiple Sclerosis"
)

# ──────────────────────────────────────────────────────────────
# 1. 인자 파서
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="vLLM chatting 기반 문제 풀이 생성")
    p.add_argument("--hf_token",     type=str, required=True)
    p.add_argument("--model_path",   type=str, required=True)
    p.add_argument("--gpu_id",       type=str, required=True)
    p.add_argument("--input_file",   type=str, required=True)
    p.add_argument("--output_dir",   type=str, required=True)
    p.add_argument("--repeat_count", type=int, required=True)
    p.add_argument("--temperature",  type=float, required=True)
    p.add_argument("--top_k",        type=int, required=True)
    p.add_argument("--max_tokens",   type=int, required=True)
    p.add_argument("--data_source_list", type=str, help="쉼표로 구분된 처리할 데이터 소스 목록")
    return p.parse_args()

# ──────────────────────────────────────────────────────────────
# 2-A. 전처리 (문제 텍스트 조립)
# ──────────────────────────────────────────────────────────────
def format_question(qdata):
    ds       = qdata["data_source"].lower()
    qid      = qdata["question_id"]
    question = qdata["question"].strip()
    orig_ans = qdata["correct_answer"].strip()              # <<< 변경: 원본 correct_answer 보존
    opts     = qdata.get("options", [])                     # <<< 변경: 원본 options 보존
    related_docs = qdata.get("related_docs", [])            # <<< 추가: related_docs 필드 가져오기

    if ds in MULTIPLE_CHOICE_SOURCES and opts:
        # 옵션에 (A),(B)... 라벨 붙이기
        labels = [f"({c})" for c in string.ascii_uppercase[:len(opts)]]
        opts_text = "\n".join(f"{lab} {opt}" for lab, opt in zip(labels, opts))
        full_q = f"{question}\n\n{opts_text}"
        gt     = orig_ans.upper()                             # <<< 변경: MC 정답 알파벳
    else:
        # 주관식은 질문만
        full_q = question
        gt     = orig_ans                                   # <<< 변경: open 정답 텍스트 그대로

    return {
        "question_id":          qid,
        "data_source":          ds,                         # <<< 변경: data_source 보존
        "question":             full_q,
        "options":              opts,                       # <<< 변경: options 리스트 보존
        "correct_answer":       orig_ans,                   # <<< 변경: 원본 correct_answer 보존
        "ground_truth_for_eval":gt,                         # <<< 변경: 평가용 정답 구분 저장
        "related_docs":         related_docs                # <<< 추가: related_docs 필드 보존
    }


# ──────────────────────────────────────────────────────────────
# 2-B. LLM 응답 후처리 함수들
# ──────────────────────────────────────────────────────────────
STEP_PATTERN = r'(?:## )?Step \d+:'

def extract_steps_from_text(txt: str):
    """## Step n: … 구간별 슬라이스"""
    mts = list(re.finditer(STEP_PATTERN, txt))
    if not mts:
        return [txt.strip()] if txt.strip() else []
    steps = []
    for i, m in enumerate(mts):
        start = m.start()
        end   = mts[i+1].start() if i+1 < len(mts) else len(txt)
        steps.append(txt[start:end].strip().replace("## ", ""))
    return steps

# ① 선택형 정답 추출
def extract_answer_from_text(txt: str):
    txt = txt.lower()

    # \boxed{b}
    m = re.findall(r'\\boxed\{\(?([a-z])\)?\}', txt)
    if m:
        return m[-1].upper()

    # the answer is b
    m_iter = re.finditer(r'(?:answer is|the answer is|final answer is)\s*:?\s*\(?([a-z])\)?',
                         txt, flags=re.I)
    m_list = list(m_iter)
    if m_list:
        return m_list[-1].group(1).upper()
    return None

# ② 주관식 정답 추출
def open_extract_answer_from_text(txt: str):
    ptns = [
        r"## Final Diagnosis:\s*(.*?)(?:\n|$)",
        r"Final Diagnosis:\s*(.*?)(?:\n|$)",
        r"diagnosis is\s*(.*?)(?:\.|$)"
    ]
    for p in ptns:
        m = re.search(p, txt, flags=re.I)
        if m:
            return m.group(1).strip()
    return ""

# ──────────────────────────────────────────────────────────────
# 2-C. PRM/ORM 변환
# ──────────────────────────────────────────────────────────────
def prm_process_solution(txt: str):
    no_nl = txt.replace("\n", " ")
    mts = list(re.finditer(STEP_PATTERN, no_nl))
    if not mts:
        return no_nl.strip() + " ки" if no_nl.strip() else ""
    head = no_nl[:mts[0].start()].strip()
    steps = []
    for i, m in enumerate(mts):
        start = m.start()
        end   = mts[i+1].start() if i+1 < len(mts) else len(no_nl)
        steps.append(no_nl[start:end].strip().replace("## ", "") + " ки")
    if head:
        steps.insert(0, head)
    return " ".join(steps)

def orm_process_solution(txt: str):
    return txt.replace("\n", " ") + " ки"

# ──────────────────────────────────────────────────────────────
# 3. JSON 입출력 유틸
# ──────────────────────────────────────────────────────────────
def load_json(path):   return json.load(open(path,  encoding="utf-8"))
def save_json(obj, p): json.dump(obj, open(p, "w", encoding="utf-8"),
                                 ensure_ascii=False, indent=2)

# ──────────────────────────────────────────────────────────────
# 4-A. 프롬프트 수집
# ──────────────────────────────────────────────────────────────
def collect_prompts(qdatas, n_samples):
    convs, meta = [], []
    for q in qdatas:
        sys_prompt = OPEN_SYSTEM_PROMPT if q["data_source"] in OPEN_SOURCES else SYSTEM_PROMPT
        conv_tmpl  = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": q["question"]}
        ]
        for _ in range(n_samples):
            convs.append(conv_tmpl)
            meta.append({
                "question_id":  q["question_id"],
                "ground_truth": q["ground_truth_for_eval"],
                "data_source":  q["data_source"]
            })
    return convs, meta

# ──────────────────────────────────────────────────────────────
# 4-B. vLLM 실행 헬퍼
# ──────────────────────────────────────────────────────────────
def llm_chat(llm, samp_params, convs, meta):
    outs = llm.chat(convs, samp_params)
    bucket = defaultdict(lambda: {"generated_texts": []})
    for i, o in enumerate(outs):
        qid = meta[i]["question_id"]
        bucket[qid]["generated_texts"].append(o.outputs[0].text)
    return bucket

# ──────────────────────────────────────────────────────────────
# 5. 메인 파이프라인
# ──────────────────────────────────────────────────────────────
def generate_all(qdatas, repeat_cnt, llm, samp_params):
    first_cnt = math.ceil(repeat_cnt * 1.25)

    # 1차
    convs1, meta1 = collect_prompts(qdatas, first_cnt)
    res1 = llm_chat(llm, samp_params, convs1, meta1)

    valid = defaultdict(list)
    for q in qdatas:
        qid = q["question_id"]
        for txt in res1[qid]["generated_texts"]:
            if 2 < len(extract_steps_from_text(txt)) < 10:
                valid[qid].append(txt)

    # 부족분 계산 후 2차
    need2 = [q for q in qdatas if len(valid[q["question_id"]]) < repeat_cnt]
    if need2:
        convs2, meta2 = [], []
        for q in need2:
            lack = (repeat_cnt - len(valid[q["question_id"]])) * 3
            sys_prompt = OPEN_SYSTEM_PROMPT if q["data_source"] in OPEN_SOURCES else SYSTEM_PROMPT
            tmpl = [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": q["question"]}
            ]
            for _ in range(lack):
                convs2.append(tmpl)
                meta2.append({
                    "question_id":  q["question_id"],
                    "ground_truth": q["ground_truth_for_eval"],
                    "data_source":  q["data_source"]
                })
        res2 = llm_chat(llm, samp_params, convs2, meta2)
        for q in need2:
            qid = q["question_id"]
            for txt in res2[qid]["generated_texts"]:
                if 2 < len(extract_steps_from_text(txt)) < 10:
                    valid[qid].append(txt)

    # ───────── 최종 JSON 조립 ─────────
    outputs = []
    for q in qdatas:
        qid  = q["question_id"]
        ds   = q["data_source"]
        gt   = q["ground_truth_for_eval"]
        sols = []
        for txt in valid[qid][:repeat_cnt]:
            if ds in OPEN_SOURCES:
                pred = open_extract_answer_from_text(txt)
                score = "None"
            else:
                pred  = extract_answer_from_text(txt)
                score = int(pred == gt) if pred else 0
            sols.append({
                "solution": txt,
                "prm_processed_solution": prm_process_solution(txt),
                "orm_processed_solution": orm_process_solution(txt),
                "answer": pred,
                "score": score
            })
        outputs.append({
            "question_id":    qid,
            "data_source":    ds,           # ← 추가
            "question":       q["question"],
            "correct_answer": gt,
            "solutions":      sols,
            "related_docs":   q["related_docs"],  # ← 수정: related_docs 필드 추가
        })
    return outputs

# ──────────────────────────────────────────────────────────────
# 6. 메인 함수
# ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    login(args.hf_token)
    print("✅  Hugging Face login 완료")

    llm = LLM(
        model=args.model_path,
        device="cuda",
        dtype="bfloat16",
        max_model_len=args.max_tokens,
        gpu_memory_utilization=0.95,
    )
    samp = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        max_tokens=args.max_tokens
    )

    raw = load_json(args.input_file)
    print(f"📂 로드: {len(raw)} 문제")

    # 특정 데이터 소스만 처리
    data_source_tag = "all"
    if args.data_source_list:
        data_sources = [ds.strip() for ds in args.data_source_list.split(',')]
        filtered_raw = [q for q in raw if q["data_source"].lower() in data_sources]
        print(f"🔍 필터링: {len(filtered_raw)}/{len(raw)} 문제 (데이터 소스: {', '.join(data_sources)})")
        raw = filtered_raw
        data_source_tag = data_sources[0]  # 첫 번째 데이터 소스만 파일명에 사용

    transformed = [format_question(q) for q in raw]

    dataset = generate_all(transformed, args.repeat_count, llm, samp)

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 모델명 추출 (경로에서 마지막 부분만)
    model_name = os.path.basename(args.model_path.rstrip('/'))
    
    # 파일명 동적 생성
    base = os.path.splitext(os.path.basename(args.input_file))[0]
    out_path = os.path.join(args.output_dir, f"{base}_{model_name}_{data_source_tag}_{args.repeat_count}.json")
    save_json(dataset, out_path)
    print(f"🎉  Done → {out_path}")

if __name__ == "__main__":
    main()
