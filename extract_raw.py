import argparse
import json
import csv
import re
from collections import Counter

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
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model alias (e.g., llama3, deepseek)")
parser.add_argument("--prompt", type=str, choices=PROMPTS_CONFIGS.keys(), required=True, help="Prompt alias (e.g., e1")
parser.add_argument("--d", type=str, default="d", help="demo or test")
args = parser.parse_args()

select_prompt = PROMPTS_CONFIGS[args.prompt]
dt_judge = args.d

if dt_judge == "d":
    input_path = f"outputs/raw_data/demo/{args.model}/{args.model}_outputs_{select_prompt}.jsonl"
    output_path = f"outputs/extract_data/demo/{args.model}/{args.model}_extracted_{select_prompt}.csv"
elif dt_judge == "t":
    input_path = f"outputs/raw_data/{args.model}/{args.model}_outputs_{select_prompt}.jsonl"
    output_path = f"outputs/extract_data/{args.model}/{args.model}_extracted_{select_prompt}.csv"

def extract_answer(text):
    """
    柔軟な形式の 'Answer: Drug A\nReason: ...' からAnswerとReasonを抽出
    - Answer: の前に複数の改行、空白、**, -- 等があっても対応
    - Reason: はテキスト全体から一行で抽出
    """
    # 改行を統一
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Answer を抽出（前に改行・スペース・**・--などが複数あっても対応）
    answer_match = re.search(r"(?:[\s\*\-]*\n)*[\s\*\-]*(?:Answer:\s*)?(Option A|Option B)?", normalized_text, re.IGNORECASE)
    # Reason を抽出（改行をスペースに）
    # reason_text = normalized_text.replace("\n", " ")
    # reason_match = re.search(r"Reason:\s*(.*)", reason_text, re.IGNORECASE)
    # 結果整形
    answer = answer_match.group(1).strip() if answer_match and answer_match.group(1) else ""
    # reason = reason_match.group(1).strip() if reason_match else ""

    return answer#, reason

def get_answer_meaning(template, answer_choice):
    """
    answer_meaningをtemplateとanswer_choiceから判定
    """
    if template == "forward":
        return "risky" if answer_choice == "Option A" else "certain" if answer_choice == "Option B" else ""
    elif template == "reversed":
        return "certain" if answer_choice == "Option A" else "risky" if answer_choice == "Option B" else ""
    return ""

# 選択数カウント用
answer_counter = Counter()
# templateごとのOption A/B選択数を集計
template_counter = {
    "forward": Counter(),
    "reversed": Counter(),
}

with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", newline="", encoding="utf-8") as outfile:

    writer = csv.writer(outfile, quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow([
        "case_id", "story", "persona", "socio_demo", 
        "p", "risky_reward", "certain_reward", 
        "template", "model",
        "answer_choice", "answer_meaning"
    ])

    for line in infile:
        record = json.loads(line)
        output_text = record.get("output", "")
        answer_choice = extract_answer(output_text)
        answer_meaning = get_answer_meaning(record.get("template", ""), answer_choice)
        answer_counter[answer_choice] += 1
        answer_counter[answer_meaning] += 1
        line_temp = record.get("template")
        line_choice = answer_choice
        if line_temp in template_counter:
            template_counter[line_temp][line_choice] += 1

        writer.writerow([
            record.get("case_id"),
            record.get("story"),
            record.get("persona"),
            record.get("socio_demo"),
            record.get("p"),
            record.get("risky_reward"),
            record.get("certain_reward"),
            record.get("template"),
            record.get("model"),
            answer_choice,
            answer_meaning,
            #reason
        ])

# 結果表示
print("✅ answer_choiceの選択状況:")
print(f"  Option A を選んだ数: {answer_counter['Option A']}")
print(f"  Option B を選んだ数: {answer_counter['Option B']}")
print(f"  choiceの不明な/空白の回答数: {answer_counter['']}")
print("\n✅ answer_meaningの選択状況:")
print(f"  risky を選んだ数: {answer_counter['risky']}")
print(f"  certain を選んだ数: {answer_counter['certain']}")
print(f"  meanの不明な/空白の回答数: {answer_counter['']}")

print("\n✅ templateごとのOption選択状況:")
for template_type in ["forward", "reversed"]:
    print(f"  - {template_type}:")
    print(f"      Option A: {template_counter[template_type]['Option A']}")
    print(f"      Option B: {template_counter[template_type]['Option B']}")
    print(f"      不明/空白: {template_counter[template_type]['']}")