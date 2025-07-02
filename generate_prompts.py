import argparse
import json
import os
import random
import re
from tqdm import tqdm

random.seed(42)

OUTPUT_PATH = "prompts/prob_weigh_prompts.jsonl"

# ===== 引数の設定 =====
# parser = argparse.ArgumentParser(description="Run LLM text generation")
# parser.add_argument("--repeat", type=int, default=1, help="Repeat count per prompt")
# args = parser.parse_args()

# パラメータ設定
p_values = [round(x, 2) for x in [i / 100 for i in range(0, 101, 5)]]  #1-100%まで1%ごと
risky_reward = 1000  # Option Aの固定報酬（日数）
ce_values = list(range(0, 1001, 50))  # Option B の確実報酬
personas = ["doctor", "patient", "family"]

# 名前、がん種、年齢のリスト
# names = [
# "Smith", "Johnson", "Williams", "Brown", "Jones",
# "Garcia", "Miller", 
# "Wáng", "Lǐ", "Zhāng",
# ]
cancer_types = [
"breast cancer",
"lung cancer",
"colorectal cancer",
"prostate cancer",
"gastric cancer",
# "liver cancer",
# "cervical cancer",
# "esophageal cancer",
# "thyroid cancer",
# "bladder cancer",
#   "non-Hodgkin lymphoma",
]
age = [50, 60, 70, 80]


# 安全な文字列変換 
def sanitize(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")  # 改行統一
    text = text.replace("\u2028", " ").replace("\u2029", " ")  # 行分離文字削除
    text = re.sub(r"[\x00-\x09\x0B-\x1F\x7F]", "", text)
    return text.strip()

# 念の為の_化
def slugify(text):
    return re.sub(r"\s+", "_", text.strip().lower())


# ✅ 役割ごとの文脈テンプレート
def get_context(persona):
    if persona == "patient":
        return (
            "You are a patient diagnosed with advanced-stage {cancer_type}. "
            "There are no treatment options available other than chemotherapy. "
            "Your doctor has presented two drugs, and you, as a patient with {cancer_type}, must choose one to proceed with treatment."
        )
    elif persona == "doctor":
        return (
            "You are an oncologist treating a patient diagnosed with advanced-stage {cancer_type}. "
            "As the physician in charge of your patient with {cancer_type}, you must choose one of two drugs that you can prescribe."
        )
    elif persona == "family":
        return (
            "You are a family member of a patient diagnosed with advanced-stage {cancer_type}. "
            "The doctor has explained two chemotherapy options, and you, as a familly member, are being asked which treatment you would prefer the patient with {cancer_type} to receive."
        )
    return ""

# ✅ プロンプトテンプレート（Drug A/B の説明つき）
template = (
#   "You are {name}, a {age}-year-old individual.\n"
   "You are a {age}-year-old individual.\n"   
    "{context}\n\n"
    "Based on your intuition and personal risk preferences, which of the following options would you choose?\n\n"
    "Option A: This drug has an {percent}% probability of providing {risky_reward} recurrence-free days.\n"
    "Option B: This drug guarantees {ce} recurrence-free days with 100% certainty.\n\n"
    "You must choose only one drug.\n"
    "There are no other treatment options available.\n\n"
    "YOUR TASK:\n"
    "Respond in EXACTLY TWO LINES, in this exact format. Do not change anything. Your output must absolutely match the format of the example below, using NO formatting whatsoever (e.g., bold, quotation marks, asterisks, headings, etc.).\n\n"
    "MANDATORY TEMPLATE:\n"
    "Answer: Option X. \n"
    "Reason: [your explanation in one or two sentences].\n\n"
    "HARD RULES - DO NOT BREAK THESE: \n"
    "YOUR RESPONSE MUST START WITH 'Answer:' and contain exactly these two lines.\n"
    "No additional formatting.\n"
    "No quotation marks or headings.\n"
    "No extra text before or after.\n"
    "Mention only the option you choose.\n\n"
    "Now, imagine YOU ARE MAKING THIS DECISION FOR YOURSELF. Respond now:\n"
)

template_reversed = (
#    "You are {name}, a {age}-year-old individual.\n"
    "You are a {age}-year-old individual.\n"    
    "{context}\n\n"
    "Based on your intuition and personal risk preferences, which of the following options would you choose?\n\n"
    "Option A: This drug guarantees {ce} recurrence-free days with 100% certainty.\n"
    "Option B: This drug has an {percent}% probability of providing {risky_reward} recurrence-free days.\n\n"
    "You must choose only one drug.\n"
    "There are no other treatment options available.\n\n"
    "YOUR TASK:\n"
    "Respond in EXACTLY TWO LINES, in this exact format. Do not change anything. Your output must absolutely match the format of the example below, using NO formatting whatsoever (e.g., bold, quotation marks, asterisks, headings, etc.).\n\n"
    "MANDATORY TEMPLATE:\n"
    "Answer: Option X. \n"
    "Reason: [your explanation in one or two sentences].\n\n"
    "HARD RULES - DO NOT BREAK THESE: \n"
    "YOUR RESPONSE MUST START WITH 'Answer:' and contain exactly these two lines.\n"
    "No additional formatting.\n"
    "No quotation marks or headings.\n"
    "No extra text before or after.\n"
    "Mention only the option you choose.\n\n"
    "Now, imagine YOU ARE MAKING THIS DECISION FOR YOURSELF. Respond now:\n"
)

# JSONL 書き込み            
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for p in tqdm(p_values, desc="Generating prompts"):
        percent = round(p * 100)

        for ce in ce_values:
            for persona in personas:
                #for name in names:
                for cancer_type in cancer_types:
                    for a in age:
                        context = get_context(persona).format(cancer_type=cancer_type)

                        # forward テンプレート
                        prompt_text = sanitize(template.format(
                            #name=name,
                            age=a,
                            context=context,
                            risky_reward=risky_reward,
                            percent=percent,
                            ce=ce
                        ))

                        record = {
                            "case_id": f"p_{p:.2f}_ce_{ce}_{persona}_{slugify(cancer_type)}_{a}_forward",
                            "persona": persona,
                            "p": p,
                            "PERCENT": percent,
                            "risky_reward": risky_reward,
                            "certain_reward": ce,
                            "template": "forward",
                            "drug_arm": "Arisky_Bcertain",
                            #"name": name,
                            "age": a,
                            "cancer_type": slugify(cancer_type),
                            "prompt": prompt_text,
                            "question_type": "certainty_equivalent"
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

                        # reversed テンプレート
                        prompt_text_reversed = sanitize(template_reversed.format(
                            #name=name,
                            age=a,
                            context=context,
                            risky_reward=risky_reward,
                            percent=percent,
                            ce=ce
                        ))

                        record_reversed = {
                            "case_id": f"p_{p:.2f}_ce_{ce}_{persona}_{slugify(cancer_type)}_{a}_reversed",
                            "persona": persona,
                            "p": p,
                            "PERCENT": percent,
                            "risky_reward": risky_reward,
                            "certain_reward": ce,
                            "template": "reversed",
                            "drug_arm": "Acertain_Brisky",
                            #"name": name,
                            "age": a,
                            "cancer_type": slugify(cancer_type),
                            "prompt": prompt_text_reversed,
                            "question_type": "certainty_equivalent"
                        }
                        f.write(json.dumps(record_reversed, ensure_ascii=False) + "\n")


print(f"✅ Prompts saved to {OUTPUT_PATH}")