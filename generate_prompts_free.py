import argparse
import json
import os
import random
import re
from tqdm import tqdm

random.seed(42)

OUTPUT_PATH = "prompts/prob_weigh_prompts_free.jsonl"

# ===== 引数の設定 =====
parser = argparse.ArgumentParser(description="Run LLM text generation")
parser.add_argument("--repeat", type=int, default=1, help="Repeat count per prompt")
args = parser.parse_args()

# パラメータ設定
p_values = [round(x, 2) for x in [i / 100 for i in range(1, 100, 20)]]  #1-100%まで1%ごと
risky_reward = 1000  # Option Aの固定報酬（日数）
personas = ["doctor", "patient", "family"]

# 患者背景パラメーター (名字、性別、年齢, がん種)
def generate_identity():
    names = [
    "Smith", "Johnson", "Williams", "Brown", "Jones",
    "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
    "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris",
    "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
    "Walker", "Young", "Allen", "King", "Wright",
    "Scott", "Torres", "Nguyen", "Hill", "Flores",
    "Green", "Adams", "Nelson", "Baker", "Hall",
    "Rivera", "Campbell", "Mitchell", "Carter", "Roberts"
    ]
    genders = ["Mr. ", "Ms. "]
    cancer_types = [
    "lung cancer",
    "breast cancer",
    "colorectal cancer",
    "liver cancer",
    "thyroid cancer",
    "cervical cancer",
    "bladder cancer",
    "non-Hodgkin lymphoma",
    "esophageal cancer",
    "pancreatic cancer",
    "leukemia",
    "kidney cancer",
    "endometrial cancer",
    "oral cancer",
    "melanoma",
    "laryngeal cancer"
    ]
    name = random.choice(names)
    gender = random.choice(genders)
    age = random.randint(50, 80)
    cancer_type = random.choice(cancer_types)
    return name, age, gender, cancer_type

# 安全な文字列変換 
def sanitize(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")  # 改行統一
    text = text.replace("\u2028", " ").replace("\u2029", " ")  # 行分離文字削除
    text = re.sub(r"[\x00-\x09\x0B-\x1F\x7F]", "", text)
    return text.strip()


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
    "You are {gender}{name}, a {age}-year-old individual.\n"
    "{context}\n\n"
    "One drug has a {percent}% probability of providing {risky_reward} recurrence-free days. The other drug guarantees a fixed number of recurrence-free days with 100% certainty.\n"
    "How many guaranteed recurrence-free days would make the two treatments equally attractive to you?\n\n"
    "YOUR TASK:\n"
    "Determine how many recurrence-free days the guaranteed treatment must offer for you to consider it equally valuable as the probabilistic one.\n\n"
    "Example output:\n"
    "Answer: XXX days. \n"
    "Reason: [your explanation in one or two sentences].\n\n"
    "HARD RULES - DO NOT BREAK THESE: \n"
    "Mention only the option you choose.\n"
    "No additional formatting.\n"
    "No quotation marks or headings.\n"
    "No extra text before or after.\n"
    "Your response must start with 'Answer:' and contain exactly these two lines.\n\n"
    "Now, imagine YOU ARE MAKING THIS DECISION FOR YOURSELF. Respond now:\n"
)

# JSONL 書き込み            
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for idx_repeat in range(1, args.repeat +1):
        for p in tqdm(p_values, desc="Generating prompts"):
            percent = round(p * 100)

#            for ce in ce_values:
            for persona in personas:
                name, age, gender, cancer_type = generate_identity()
                context = get_context(persona)
                context = context.format(cancer_type=cancer_type)

                # 通常順
                prompt_text = sanitize(template.format(
                    gender=gender,
                    name=name,
                    age=age,
                    context=context,
                    risky_reward=risky_reward,
                    percent=percent,
                ))
                record = {
                    "case_id": f"p_{p:.2f}_ce_{persona}_forward",
                    "persona": persona,
                    "p": p,
                    "PERCENT": percent,
                    "risky_reward": risky_reward,
                    "name": name,
                    "age": age,
                    "gender": gender,
                    "cancer_type": cancer_type,
                    "prompt": prompt_text,
                    "question_type": "certainty_equivalent"
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"✅ Prompts saved to {OUTPUT_PATH}")