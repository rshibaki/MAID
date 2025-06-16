import argparse
import json
import os
import numpy as np
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from openai import OpenAI
from together import Together

# =====モデルマッピング（エイリアス → 実IDと出力パス）=====
MODEL_CONFIGS = {
    "llama3": {
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "output_path": "outputs/raw_data/llama3_outputs.jsonl"
    },
    "deepseek": {
        "model_id": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "output_path": "outputs/raw_data/deepseek_outputs.jsonl"
    },
    "gemma3": {
        "model_id": "google/gemma-7b-it",
        "output_path": "outputs/raw_data/gemma3_outputs.jsonl"
    },
    "gpt4.1-mini": {
        "model_id": "gpt-4.1-mini",
        "output_path": "outputs/raw_data/gpt41_mini_outputs.jsonl"
    },
    "gpt4.1": {
        "model_id": "gpt-4.1",
        "output_path": "outputs/raw_data/gpt41_outputs.jsonl"
    },
    "llama4": {
        "model_id": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "output_path": "outputs/raw_data/llama4_outputs.jsonl"
    }
}

# ===== 引数の設定 =====
parser = argparse.ArgumentParser(description="Run LLM text generation")
parser.add_argument("--model", type=str, choices=MODEL_CONFIGS.keys(), required=True, help="Model alias (e.g., llama3, deepseek, gpt4.1-mini)")
# parser.add_argument("--repeat", type=int, default=1, help="Repeat count per prompt")
args = parser.parse_args()

config = MODEL_CONFIGS[args.model]
model_id = config["model_id"]
output_path = config["output_path"]

# 入出力ファイル
PROMPT_PATH = "prompts/prob_weigh_prompts.jsonl"

# シード固定
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ===== モデル読み込みと生成器設定 =====
if args.model in ["gpt4.1-mini", "gpt4.1"]:
    # OpenAI API クライアント
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    use_openai = True
    use_together = False
elif args.model.startswith("llama4"):
    # Together用
    together_client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    use_openai = False
    use_together = True
else:
    # transformers モデル読み込み
    print(f"🔄 Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        token=True
    )
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    use_openai = False

# プロンプト読み込み
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    original_prompts = [json.loads(line) for line in f]

# 出力ファイルオープン
with open(output_path, "w", encoding="utf-8") as out_f:
    prompts = original_prompts.copy()
    random.shuffle(prompts)

    for entry in tqdm(prompts, desc=f"Run prompts"):
        prompt_text = entry["prompt"]
        case_id = entry["case_id"]

        try:
            if use_openai:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=100,
                    temperature=0.7,
                )
                output_text = response.choices[0].message.content
            elif use_together:
                response = together_client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=100,
                    temperature=0.7,
                )
                output_text = response.choices[0].message.content
            else:
                result = generator(
                    prompt_text,
                    max_new_tokens=100,
                    do_sample=False,
                    return_full_text=False,
                )
                output_text = result[0]["generated_text"]

            record = {
                **entry,
                "model": model_id,
                "output": output_text
            }
            out_f.write(json.dumps(record) + "\n")

        except Exception as e:
            print(f"❌ Error on {case_id} (run: {e}")

print(f"✅ All results saved to {output_path}")
