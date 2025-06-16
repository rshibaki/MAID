import argparse
import json
import csv
import re
from collections import Counter


# ===== 引数の設定 =====
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model alias (e.g., llama3, deepseek)")
args = parser.parse_args()

input_path = f"outputs/raw_data/{args.model}_outputs.jsonl"
output_path = f"outputs/extract_data/{args.model}_extracted.csv"

def extract_answer(text):
    """
    柔軟な形式の 'Answer: Drug A\nReason: ...' からAnswerとReasonを抽出
    - Answer: の前に複数の改行、空白、**, -- 等があっても対応
    - Reason: はテキスト全体から一行で抽出
    """
    # 改行を統一
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Answer を抽出（前に改行・スペース・**・--などが複数あっても対応）
    answer_match = re.search(r"(?:[\s\*\-]*\n)*[\s\*\-]*Answer:\s*(Option A|Option B)?", normalized_text, re.IGNORECASE)
    # Reason を抽出（改行をスペースに）
    reason_text = normalized_text.replace("\n", " ")
    reason_match = re.search(r"Reason:\s*(.*)", reason_text, re.IGNORECASE)
    # 結果整形
    answer = answer_match.group(1).strip() if answer_match and answer_match.group(1) else ""
    reason = reason_match.group(1).strip() if reason_match else ""

    return answer, reason

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

with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", newline="", encoding="utf-8") as outfile:

    writer = csv.writer(outfile)
    writer.writerow([
        "case_id", "persona", "p", "risky_reward", "certain_reward", "model",
        "template", "option_arm",
        "answer_choice", "answer_meaning", "reason"
    ])

    for line in infile:
        record = json.loads(line)
        output_text = record.get("output", "")
        answer_choice, reason = extract_answer(output_text)
        answer_meaning = get_answer_meaning(record.get("template", ""), answer_choice)
        answer_counter[answer_choice] += 1
        answer_counter[answer_meaning] += 1

        writer.writerow([
            record.get("case_id"),
            record.get("persona"),
            record.get("p"),
            record.get("risky_reward"),
            record.get("certain_reward"),
            record.get("model"),
            record.get("template"),
            record.get("option_arm"),
            answer_choice,
            answer_meaning,
            reason
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

# # 空白回答だけを抽出して保存
# blank_output_path = "trytest/llama3_blank_answers.jsonl"

# with open(output_path, "r", encoding="utf-8") as csvfile, \
#      open(blank_output_path, "w", encoding="utf-8") as blank_outfile:
    
#     reader = csv.DictReader(csvfile)
    
#     for row in reader:
#         if not row["answer_choice"] or row["answer_choice"].strip() == "":
#             blank_record = {
#                 "case_id": row["case_id"],
#                 "persona": row["persona"],
#                 "p": row["p"],
#                 "risky_reward": row["risky_reward"],
#                 "certain_reward": row["certain_reward"],
#                 "model": row["model"],
#                 "template": row["template"],
#                 "drug_arm": row["drug_arm"],
#                 "answer_choice": row["answer_choice"],
#                 "answer_mean": row["answer_mean"],
#                 "reason": row["reason"]
#             }
#             blank_outfile.write(json.dumps(blank_record, ensure_ascii=False) + "\n")

# print(f"📝 空白回答のレコードを {blank_output_path} に保存しました。")



# with open("outputs/raw_data/llama3_outputs_10x.jsonl", "r", encoding="utf-8") as f:
#     lines = f.readlines()
#     print(f"📦 JSONLファイルのレコード数: {len(lines)}")

# with open("outputs/parsed_llama3_answers.csv", "r", encoding="utf-8") as f:
#     rows = list(csv.reader(f))
#     print(f"📄 CSVファイルのデータ行数（ヘッダー除く）: {len(rows) - 1}")
    
# with open("trytest/llama3_blank_answers.jsonl", "r", encoding="utf-8") as f:
#     lines = f.readlines()
#     print(f"📦 blank_JSONLファイルのレコード数: {len(lines)}")