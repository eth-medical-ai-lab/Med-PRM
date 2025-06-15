# download_hf_repos.py

import os
import json
from huggingface_hub import snapshot_download

def download_and_inspect(repo_id: str, repo_type: str, base_dir: str):
    # repo_id 마지막 부분을 폴더명으로 사용
    name = repo_id.split("/")[-1]
    local_dir = os.path.join(base_dir, name)
    os.makedirs(local_dir, exist_ok=True)

    print(f"\n▶ Downloading {repo_id!r} into {local_dir!r} …")
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=local_dir
    )
    files = sorted(os.listdir(local_dir))
    print("  • Files in", name, ":", files)

    # JSON 검사 (dataset일 경우)
    for fname in files:
        if not fname.endswith(".json") or fname.endswith(".index.json"):
            continue

        fpath = os.path.join(local_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"    – Failed to load JSON from {fname}: {e}")
                break

        # 데이터가 list인지 dict인지 구분
        if isinstance(data, list):
            print(f"    – Loaded JSONL/list of length {len(data)}")
            obj = data[0]
        elif isinstance(data, dict):
            print(f"    – Loaded JSON object with keys: {list(data.keys())}")
            obj = data
        else:
            print(f"    – Unknown JSON structure in {fname}")
            break

        # 공통으로 첫 레코드(혹은 객체)의 키 보여주기
        print("    – Keys of first record/object:", list(obj.keys()))
        break

if __name__ == "__main__":

    # 3) Reward model → ../model_train/llama-3.1-medprm-reward-v1.0/
    download_and_inspect(
        repo_id="dmis-lab/llama-3.1-medprm-reward-v1.0",
        repo_type="model",
        base_dir="model_train"
    )

    # 1) Training set → ../dataset/dataset_1_train-dataset/
    download_and_inspect(
        repo_id="dmis-lab/llama-3.1-medprm-reward-training-set",
        repo_type="dataset",
        base_dir="dataset/dataset_1_train_dataset"
    )

    # 2) Test set → ../dataset/dataset_4_scored-dataset/
    download_and_inspect(
        repo_id="dmis-lab/llama-3.1-medprm-reward-test-set",
        repo_type="dataset",
        base_dir="dataset/dataset_3_sampled_dataset"
    )

