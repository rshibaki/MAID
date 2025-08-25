# ===== これより上は「環境変数セット」だけ。torch/vllm より前 =====
import os, multiprocessing as mp
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["NVIDIA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
#os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"  # 公式の切替

try:
    mp.set_start_method("spawn", force=True)  # ← “必ず” import torch より前
except RuntimeError:
    pass

# ここから下で初めて import
import argparse
import json
import random
import logging
import numpy as np
import torch

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from vllm import LLM, SamplingParams
from vllm.config import DeviceConfig
from tqdm import tqdm        
from openai import OpenAI    
from together import Together

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
        # "model_id": "Qwen/Qwen2.5-72B-Instruct",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
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

def build_vllm(model_id: str) -> LLM:
    tp = int(os.getenv("TP_SIZE", str(torch.cuda.device_count() or 1)))

    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        dtype="bfloat16",                 # L40S なので BF16 推奨（FP16 でも可）
        tensor_parallel_size=tp,          # ← 1 → 4（自動）
        pipeline_parallel_size=1,
        gpu_memory_utilization=0.82,      # ← 0.72 → 0.90 に上げる
        max_model_len=1024,            # ← 72B ではまず 4k くらいから
        enforce_eager=True,
        max_num_seqs=1,
        kv_cache_dtype="auto",
        disable_custom_all_reduce=True,
        download_dir=os.getenv("HF_HOME", "/workspace/hf_cache"),
    )
    return llm


def main():
    # ===== 引数の設定 =====
    parser = argparse.ArgumentParser(description="Run LLM text generation")
    parser.add_argument("--model", type=str, choices=MODEL_CONFIGS.keys(), required=True, help="Model alias (e.g., llama3, deepseek, gpt4.1-mini)")
    parser.add_argument("--prompt", type=str, choices=PROMPTS_CONFIGS.keys(), required=True, help="Prompt alias (e.g., e1")
    parser.add_argument("--temp", type=float, default=0.3, help="Temperature (default: 0.4)")
    parser.add_argument("--d", type=str, default="d", help="demo or test")
    parser.add_argument("--engine", type=str, choices=["auto", "vllm", "hf", "openai", "together"], default="vllm", help="Inference engine")
    args = parser.parse_args()

    # ログイン設定
    config = MODEL_CONFIGS[args.model]
    model_id = config["model_id"]
    # output_path = config["output_path"]
    temperature = args.temp
    select_prompt = PROMPTS_CONFIGS[args.prompt]

    if args.d == "d":
        # 入力ファイル
        PROMPT_PATH = f"prompts/demo/prompts_{select_prompt}.jsonl"
        # 出力パス
        output_path = f"outputs/raw_data/demo/{args.model}/{args.model}_outputs_{select_prompt}.jsonl"
    elif args.d == "t":
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
        
    # 事前チェック（spawn/可視GPU）
    print("mp start_method =", mp.get_start_method(allow_none=True))
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("cuda_count =", torch.cuda.device_count())

    # ===== モデル読み込みと生成器設定 =====
    use_openai = False
    use_together = False
    use_vllm = False
    use_hf = False

    # OpenAI / Together は明示的に固定
    if args.engine == "openai" or args.model in ["gpt41mini", "gpt4.1"]:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        use_openai = True

    elif args.engine == "together" or args.model.startswith("llama4"):
        from together import Together
        together_client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        use_together = True

    else:
        # vLLM を優先（指定 or auto で vLLM が使えれば）
        if (args.engine in ["auto", "vllm"]):
            use_vllm = True
        elif args.engine == "hf":
            use_hf = True


    hf_model = None
    tokenizer = None
    vllm_engine = None
    vllm_tokenizer = None  # chat_template を使いたい場合に使う

    if use_openai or use_together:
        pass  # 何もロードしない (API 経由)

    elif use_vllm:
        # vLLM 初期化
        def _tp_size():
            try:
                return torch.cuda.device_count()
            except Exception:
                return 1

        logging.info(f"🔄 Loading model with vLLM: {model_id}")
        vllm_engine = build_vllm(model_id)
        # 必要に応じて chat template を使う準備（無くてもOK）
        try:
            from transformers import AutoTokenizer as _HFAT
            vllm_tokenizer = _HFAT.from_pretrained(model_id, trust_remote_code=True)
        except Exception:
            vllm_tokenizer = None

    elif use_hf:
        from transformers import BitsAndBytesConfig
        # transformers でロード（従来どおり）
        def auto_max_memory(ratio=0.90):
            mm = {}
            for i in range(torch.cuda.device_count()):
                total = torch.cuda.get_device_properties(i).total_memory
                cap = int(total * ratio) // (1024**3)
                mm[i] = f"{cap}GiB"
            mm["cpu"] = "96GiB"
            return mm

        logging.info(f"🔄 Loading (HF) model: {model_id}")
        
        # from transformers.utils import logging as hf_logging
        # hf_logging.set_verbosity_info()  # 進捗を少し詳しく出す

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            padding_side="left",
        )
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        max_mem = auto_max_memory(0.90)
        
        use_flash = False
        try:
            import flash_attn  # あるか確認
            use_flash = True
        except Exception:
            pass

        attn_impl = "flash_attention_2" if use_flash else "sdpa"
        
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory=max_mem,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            #token=hf_token,
            attn_implementation=attn_impl,
            quantization_config=quant_config,
        )
        hf_model.eval()
        try:
            from accelerate import infer_auto_device_map
            print(getattr(hf_model, "hf_device_map", "no device map"))
        except Exception:
            pass




    # ===== 🔧 ミニバッチ推論を導入 =====
    batch_size = 32
    #l40s: qwen2.5-70B(32)
    max_tokens = 20

    from torch import inference_mode
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f]
    random.shuffle(prompts)
    with open(output_path, "w", encoding="utf-8") as out_f:


    # with open(output_path, "w", encoding="utf-8") as out_f:
    #     prompts = original_prompts.copy()
    #     random.shuffle(prompts)

        # OpenAI
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

        # Together
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

        # vLLM
        elif use_vllm:
            # vLLM はバッチを自動で捌けるので、リストで一気に投げてもOK
            # bad_words_ids は無いので “停止語” で近似（出力の途中停止）
            stop_words = ["HARD RULES", "INSTRUCTIONS", "<think>", "</think>", "Option X"]
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=0.90,
                top_k=50,
                max_tokens=max_tokens,
                stop=stop_words,
            )

            # 必要なら chat_template を適用（普段はそのままでもOK）
            def _render(p: str) -> str:
                if vllm_tokenizer is not None and getattr(vllm_tokenizer, "chat_template", None):
                    try:
                        return vllm_tokenizer.apply_chat_template(
                            [{"role": "user", "content": p}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    except Exception:
                        return p
                return p

            all_text_prompts = [_render(e["prompt"]) for e in prompts]

            # 生成（順序は入力順と対応）
            logging.info(f"🚀 Generating {len(all_text_prompts)} prompts via vLLM …")
            outputs = vllm_engine.generate(all_text_prompts, sampling_params)

            # 保存
            for entry, out in zip(prompts, outputs):
                try:
                    # 最上位ビーム（またはサンプル）を採用
                    text = out.outputs[0].text if out.outputs else ""
                    record = {**entry, "model": model_id, "output": text.strip()}
                    out_f.write(json.dumps(record) + "\n")
                except Exception as e:
                    print(f"❌ Error on {entry.get('case_id','N/A')} (save: {e})")

        # HF (従来どおり)
        else:
            with inference_mode():
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


if __name__ == "__main__":
    main()