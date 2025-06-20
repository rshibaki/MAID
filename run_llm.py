import argparse
import json
import os
import numpy as np
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig # pipeline
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
parser.add_argument("--temp", type=float, default=0.7, help="Temperature (default: 0.7)")
args = parser.parse_args()

config = MODEL_CONFIGS[args.model]
model_id = config["model_id"]
output_path = config["output_path"]

# 入出力ファイル
PROMPT_PATH = "prompts/prob_weigh_prompts.jsonl"
temperature = args.temp

# シード固定
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ===== モデル読み込みと生成器設定 =====
use_openai = False
use_together = False

if args.model in ["gpt4.1-mini", "gpt4.1"]:
    # OpenAI API クライアント
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    use_openai = True
elif args.model.startswith("llama4"):
    # Together用
    together_client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
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
    model.eval()

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
                    max_tokens=50,
                    temperature=temperature,
                )
                output_text = response.choices[0].message.content
            elif use_together:
                response = together_client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=50,
                    temperature=temperature,
                )
                output_text = response.choices[0].message.content
            else:
                # 🔧 修正：transformers で generate() を使用
                inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

                generation_config = GenerationConfig(
                    max_new_tokens=50,
                    temperature=temperature,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

                output_ids = model.generate(
                    **inputs,
                    generation_config=generation_config
                )

                output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            record = {
                **entry,
                "model": model_id,
                "output": output_text
            }
            out_f.write(json.dumps(record) + "\n")

        except Exception as e:
            print(f"❌ Error on {case_id} (run: {e}")

print(f"✅ All results saved to {output_path}")
