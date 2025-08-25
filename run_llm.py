import argparse
import json
import os
import random
import logging
import numpy as np
import torch

from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from openai import OpenAI
from together import Together

from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
)

# ✅ ログ設定はここで一度だけ
logging.basicConfig(level=logging.INFO)

# この2行を追加してください
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# =====モデルマッピング（エイリアス → 実IDと出力パス）=====
MODEL_CONFIGS = {
    "llama3": {
        "model_id": "meta-llama/Llama-3.3-70B-Instruct",
        #"output_path": "outputs/raw_data/llama3_outputs_e1.jsonl"
    },
    "deepseek": {
        "model_id": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "output_path": "outputs/raw_data/deepseek_outputs.jsonl"
    },
    "gemma3": {
        "model_id": "google/gemma-3-27b-it",
        "output_path": "outputs/raw_data/gemma3_outputs.jsonl"
    },
    "gpt41mini": {
        "model_id": "gpt-4.1-mini",
        #"output_path": "outputs/raw_data/gpt41_mini_outputs.jsonl"
    },
    "gpt4.1": {
        "model_id": "gpt-4.1",
        "output_path": "outputs/raw_data/gpt41_outputs.jsonl"
    },
    "llama4": {
        "model_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct" #"meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        #"output_path": "outputs/raw_data/llama4_outputs.jsonl"
    },
    "qwen3": {
        "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        #"output_path": "outputs/raw_data/qwen3_outputs.jsonl"
    },
    "qwen2.5": {
        "model_id": "Qwen/Qwen2.5-72B-Instruct",
        #"output_path": "outputs/raw_data/qwen3_outputs.jsonl"
    },
    "gptoss": {
        "model_id": "openai/gpt-oss-120b",
    },
}

PROMPTS_CONFIGS = {
    # efficacy
    "e1": "E1_ColorectalCan_PFS",
    "e2": "E2_LiverCan_size",
    "e3": "E3_HT_BP",
    "e4": "E4_BA_peakflow",
    "e5": "E5_Alzheimers_NPI",
    # safety
    "s1": "S1_PancreaticCan_anorexia",
    "s2": "S2_SLE_infection",
    "s3": "S3_PD_dyskinesia",
    "s4": "S4_AF_bleeding",
    "s5": "S5_Influenza_fever",
    # cost
    "c1": "C1_COVID19_hospitalization",
    "c2": "C2_DM_hospitalization",
    "c3": "C3_IPF_transplant",
    "c6": "C6_Cancer_aiding",
    "c7": "C7_HCV_aiding",
    "c8": "C8_HT_medcost",
}

# ===== 引数の設定 =====
parser = argparse.ArgumentParser(description="Run LLM text generation")
parser.add_argument("--model", type=str, choices=MODEL_CONFIGS.keys(), required=True, help="Model alias (e.g., llama3, deepseek, gpt4.1-mini)")
parser.add_argument("--prompt", type=str, choices=PROMPTS_CONFIGS.keys(), required=True, help="Prompt alias (e.g., e1")
parser.add_argument("--temp", type=float, default=0.3, help="Temperature (default: 0.4)")
parser.add_argument("--d", type=str, default="d", help="demo or test")
args = parser.parse_args()

# ログイン設定
config = MODEL_CONFIGS[args.model]
model_id = config["model_id"]
# output_path = config["output_path"]
temperature = args.temp
select_prompt = PROMPTS_CONFIGS[args.prompt]
dt_judge = args.d

if dt_judge == "d":
    # 入力ファイル
    PROMPT_PATH = f"prompts/demo/prompts_{select_prompt}.jsonl"
    # 出力パス
    output_path = f"outputs/raw_data/demo/{args.model}/{args.model}_outputs_{select_prompt}.jsonl"
elif dt_judge == "t":
    # 入力ファイル
    PROMPT_PATH = f"prompts/prompts_{select_prompt}.jsonl"
    # 出力パス
    output_path = f"outputs/raw_data/{args.model}/{args.model}_outputs_{select_prompt}.jsonl"

# 起動時に自動ログイン
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

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
hf_model = None
tokenizer = None

if args.model in ["gpt41mini", "gpt4.1"]:
    # OpenAI API クライアント
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    use_openai = True
    
elif args.model.startswith("llama4"):
    # Together用
    together_client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    use_together = True

# elif args.model == "gptoss":
#     # ✅ vLLMは使わない：Transformers + Harmony で gpt-oss を実行
#     # まずは 20B で疎通 → 問題なければ 120B（envで切替）
#     model_id = os.getenv("GPTOSS_MODEL_ID", "openai/gpt-oss-20b")  # 120Bにするなら env で上書き
#     print(f"🔄 Loading GPT-OSS via Transformers: {model_id}")
#     tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side="left")
#     hf_model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         trust_remote_code=True,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",            # 4×L40S に自動シャーディング
#         low_cpu_mem_usage=True,
#         token=hf_token,
#     )
#     hf_model.eval()

#     # Harmony エンコーディング
#     harmony_enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
#     stop_token_ids = harmony_enc.stop_tokens_for_assistant_actions()
#     stop_token_ids = list(stop_token_ids)  # torch.generate に渡すため list 化

    
else:
    # transformers モデル読み込み
    print(f"🔄 Loading (quantized) model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side="left")
    
    # # 4bit量子化
    # from transformers import BitsAndBytesConfig
    # quant_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #                                   )
    
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        #quantization_config=quant_config, 4bit量子化に伴う
        # torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        token=hf_token,
    )
    hf_model.eval()
    #model = torch.compile(model)
    


# プロンプト読み込み
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    original_prompts = [json.loads(line) for line in f]

# ===== 🔧 ミニバッチ推論を導入 =====
batch_size = 32  # l40sのLlama-3.3-70B-Instructは64, Qwen3-30B-A3B-Instruct-2507は256, Qwen2.5-72B-Instructは32
max_tokens = 20

# 推論ループを inference_mode で包む
from torch import inference_mode

with open(output_path, "w", encoding="utf-8") as out_f:
    prompts = original_prompts.copy()
    random.shuffle(prompts)
    #prompts = prompts[:5]  # デバッグ用に5件だけ

    with inference_mode():
        if use_openai:
            for entry in tqdm(prompts, desc="Run prompts (OpenAI)"):
                try:
                    response = client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": entry["prompt"]}],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    output_text = response.choices[0].message.content
                    record = {**entry, "model": model_id, "output": output_text.strip()}
                    out_f.write(json.dumps(record) + "\n")
                except Exception as e:
                    print(f"❌ Error on {entry.get('case_id','N/A')} (run: {e})")

        elif use_together:
            for entry in tqdm(prompts, desc="Run prompts (Together)"):
                try:
                    response = together_client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": entry["prompt"]}],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    output_text = response.choices[0].message.content
                    record = {**entry, "model": model_id, "output": output_text.strip()}
                    out_f.write(json.dumps(record) + "\n")
                except Exception as e:
                    print(f"❌ Error on {entry.get('case_id','N/A')} (run: {e})")

        # elif args.model == "gptoss":
        #     for entry in tqdm(prompts, desc="Run prompts (gpt-oss tf)"):
        #         convo = Conversation.from_messages([
        #             Message.from_role_and_content(Role.SYSTEM,    SystemContent.new()),
        #             Message.from_role_and_content(Role.DEVELOPER, DeveloperContent.new().with_instructions("Respond concisely")),
        #             Message.from_role_and_content(Role.USER, entry["prompt"]),
        #         ])

        #         # 0.0.4 は "for_text_completion" が無い → token_ids を取得して直接渡す
        #         prefill_ids = harmony_enc.render_conversation_for_completion(convo, Role.ASSISTANT)  # List[int]

        #         # generate に token_ids をそのまま入れる
        #         input_ids = torch.tensor([prefill_ids], device=hf_model.device, dtype=torch.long)
        #         attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        #         try:
        #             outputs = hf_model.generate(
        #                 input_ids=input_ids,
        #                 attention_mask=attention_mask,
        #                 max_new_tokens=max_tokens,
        #                 do_sample=True,
        #                 temperature=temperature,
        #                 pad_token_id=tokenizer.eos_token_id,
        #                 eos_token_id=stop_token_ids,   # Harmony 停止トークン（list OK）
        #             )
        #             new_tokens = outputs[0][input_ids.shape[1]:]
        #             gen = tokenizer.decode(new_tokens, skip_special_tokens=True)
        #             record = {**entry, "model": model_id, "output": gen.strip()}
        #             out_f.write(json.dumps(record) + "\n")
        #         except Exception as e:
        #             print(f"❌ Error on {entry.get('case_id','N/A')} (run: {e})")


        else:
            # 通常の Transformers 推論（バッチ）
            for i in tqdm(range(0, len(prompts), batch_size), desc="Run prompts (HF)"):
                batch = prompts[i : i + batch_size]
                prompt_texts = [e["prompt"] for e in batch]
                inputs = tokenizer(prompt_texts, return_tensors="pt",
                                   padding=True, truncation=True).to(hf_model.device)
                
                bad_words = ["HARD RULES", "INSTRUCTIONS", "<think>", "</think>", "Option X"]
                bad_words_ids = [tokenizer(bw, add_special_tokens=False).input_ids for bw in bad_words]

                generation_config = GenerationConfig(
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.90,
                    top_k=50,
                    do_sample=True,
                    bad_words_ids=bad_words_ids,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                )

                try:
                    output_ids = hf_model.generate(**inputs, generation_config=generation_config)
                    for j, entry in enumerate(batch):
                        tokens = output_ids[j, inputs.input_ids.shape[1]:]
                        output_text = tokenizer.decode(tokens, skip_special_tokens=True)
                        record = {**entry, "model": model_id, "output": output_text.strip()}
                        out_f.write(json.dumps(record) + "\n")
                except Exception as e:
                    for entry in batch:
                        print(f"❌ Error on {entry.get('case_id','N/A')} (run: {e})")

print(f"✅ All results saved to {output_path}")