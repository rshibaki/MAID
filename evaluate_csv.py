import argparse
import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
#from io import StringIO

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

# パスの設定
if dt_judge == "d":
    input_path =f"outputs/extract_data/demo/{args.model}/{args.model}_extracted_{select_prompt}.csv"
    output_csv_path = f"outputs/evaluate/demo/{args.model}/{args.model}_evaluated_{select_prompt}.csv"
    output_figure_path = f"outputs/figures/demo/{args.model}/{args.model}_sop_{select_prompt}.png"
    output_figure_adjust_path = f"outputs/figures/demo/{args.model}/{args.model}_sop_{select_prompt}_adjust.png"
    output_figure_invert_path = f"outputs/figures/demo/{args.model}/{args.model}_sop_{select_prompt}_invert.png"
    output_figure_median_path = f"outputs/figures/demo/{args.model}/{args.model}_sop_{select_prompt}_median.png"
elif args.d == "t":
    input_path =f"outputs/extract_data/{args.model}/{args.model}_extracted_{select_prompt}.csv"
    output_csv_path = f"outputs/evaluate/{args.model}/{args.model}_evaluated_{select_prompt}.csv"
    output_figure_path = f"outputs/figures/{args.model}/{args.model}_sop_{select_prompt}.png"
    output_figure_adjust_path = f"outputs/figures/{args.model}/{args.model}_sop_{select_prompt}_adjust.png"
    output_figure_invert_path = f"outputs/figures/{args.model}/{args.model}_sop_{select_prompt}_invert.png"
    output_figure_median_path = f"outputs/figures/{args.model}/{args.model}_sop_{select_prompt}_median.png"

# CSVをDataFrameとして読み込み
df = pd.read_csv(input_path, encoding="utf-8-sig")

# # それぞれの変数を格納する
total_numb = len(df)
answer_choice_counts = df["answer_choice"].value_counts().to_dict()
count_answer_A = answer_choice_counts.get("Option A", 0)
count_answer_B = answer_choice_counts.get("Option B", 0)

# ===== 1️⃣ persona,p,certain_reward,risky_reward,total_numb のベーステーブル作成 =====
base = (
    df.groupby(["persona", "p", "certain_reward", "risky_reward"])
    .size()
    .reset_index(name="total_numb")
)

# ===== 2️⃣ answer_choice × answer_meaning のクロス集計 =====
counts = (
    df.groupby(["persona", "p", "certain_reward", "risky_reward", "answer_choice", "answer_meaning"])
    .size()
    .reset_index(name="count")
)

# ★ 修正1: pivot 後に「期待する4列」を reindex で必ず作る
expected_cols = pd.MultiIndex.from_product(
    [["Option A", "Option B"], ["certain", "risky"]],
    names=["answer_choice", "answer_meaning"],
)

pivot_counts = (
    counts.pivot_table(
        index=["persona", "p", "certain_reward", "risky_reward"],
        columns=["answer_choice", "answer_meaning"],
        values="count",
        fill_value=0,
        aggfunc="sum",
    )
    .reindex(columns=expected_cols, fill_value=0)  # ← ここがポイント
    .reset_index()
)

# カラム名整理（Multiindex→Singleへ）
pivot_counts.columns = [
    "persona",
    "p",
    "certain_reward",
    "risky_reward",
    "count_A_certain",
    "count_A_risky",
    "count_B_certain",
    "count_B_risky",
]

# ===== 3️⃣ base とマージして detailed_counts 作成 =====
detailed_counts = base.merge(
    pivot_counts,
    on=["persona", "p", "certain_reward", "risky_reward"],
    how="left",
).fillna(0)

# ===== 4️⃣ 集計列を追加 =====
detailed_counts["count_risky"] = detailed_counts["count_A_risky"] + detailed_counts["count_B_risky"]
detailed_counts["count_certain"] = detailed_counts["count_A_certain"] + detailed_counts["count_B_certain"]

detailed_counts["prob_certain"] = (
    detailed_counts["count_certain"] / (detailed_counts["count_certain"] + detailed_counts["count_risky"])
).round(3)

detailed_counts["count_risky_adjust"] = (
    (detailed_counts["count_A_risky"] * total_numb / count_answer_A)
    + (detailed_counts["count_B_risky"] * total_numb / count_answer_B)
).round(3)
detailed_counts["count_certain_adjust"] = (
    (detailed_counts["count_A_certain"] * total_numb / count_answer_A)
    + (detailed_counts["count_B_certain"] * total_numb / count_answer_B)
).round(3)
detailed_counts["prob_certain_adjust"] = (
    detailed_counts["count_certain_adjust"]
    / (detailed_counts["count_certain_adjust"] + detailed_counts["count_risky_adjust"])).round(3)

int_cols = [
    "total_numb",
    "count_A_certain",
    "count_A_risky",
    "count_B_certain",
    "count_B_risky",
    "count_risky",
    "count_certain",
]
detailed_counts[int_cols] = detailed_counts[int_cols].astype(int)

# 必要なら表示
detailed_counts.to_csv(output_csv_path, index=False)

# ===== CSV保存後に総数を表示 =====
cols_to_sum = [
    "total_numb",
    "count_A_certain",
    "count_A_risky",
    "count_B_certain",
    "count_B_risky",
    "count_risky",
    "count_certain",
]
print(f"count_answer_A: {count_answer_A}, count_answer_B: {count_answer_B}")

# 合計を計算
totals = detailed_counts[cols_to_sum].sum()

print("===== 合計値 =====")
for col, val in totals.items():
    print(f"{col}: {val}")


# Figure作成
# 読み込みと加工
df_csv = pd.read_csv(output_csv_path)
max_ce = df.iloc[0]["risky_reward"]

# 条件を満たす最小certain_reward抽出
filtered = df_csv[df_csv['prob_certain'] >= 0.5].copy()
if args.prompt.startswith("e"):
    min_rewards = filtered.groupby(['persona', 'p', "risky_reward"])['certain_reward'].min ().reset_index()
    min_rewards['certain_reward'] = min_rewards['certain_reward'].fillna(max_ce)
elif args.prompt.startswith(("s", "c")):
    min_rewards = filtered.groupby(['persona', 'p', "risky_reward"])['certain_reward'].max().reset_index()
    min_rewards['certain_reward'] = min_rewards['certain_reward'].fillna(0)
min_rewards['subjective_prob'] = min_rewards['certain_reward'] / min_rewards["risky_reward"]

plt.figure()

for persona in min_rewards['persona'].unique():
    sub_df = min_rewards[min_rewards['persona'] == persona].sort_values('p')

    x = sub_df['p'].values
    y = sub_df['subjective_prob'].values

    # LOWESSで滑らかな曲線を作成
    lowess_smoothed = lowess(y, x, frac=0.6)  # fracで滑らかさを調整 (0に近づけるほど元データ寄り、1に近づくほど滑らか)
    x_smooth = lowess_smoothed[:, 0]
    y_smooth = lowess_smoothed[:, 1]

    # 滑らかな曲線を描画
    plt.plot(x_smooth, y_smooth, label=persona)

    # 元の点も参考として描画
    plt.scatter(x, y, marker='o', alpha=0.5)

plt.plot([0, 1], [0, 1], 'k--', label='Y = X (Reference)')
plt.xlabel('Objective Probability')
plt.ylabel('Subjective Probability')
plt.title('Subjective-Objective Probability by Persona')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(title='Persona')
plt.grid(True)
plt.tight_layout()

plt.savefig(output_figure_path, dpi=300)
plt.close()

########## Adjusted Figure ##########
# 条件を満たす最小certain_reward抽出
filtered_adj = df_csv[df_csv['prob_certain_adjust'] >= 0.5].copy()
if args.prompt.startswith("e"):
    min_rewards_adj = filtered_adj.groupby(['persona', 'p', "risky_reward"])['certain_reward'].min().reset_index()
    min_rewards_adj['certain_reward'] = min_rewards_adj['certain_reward'].fillna(max_ce)
elif args.prompt.startswith(("s", "c")):
    min_rewards_adj = filtered_adj.groupby(['persona', 'p', "risky_reward"])['certain_reward'].max().reset_index()
    min_rewards_adj['certain_reward'] = min_rewards_adj['certain_reward'].fillna(0)
min_rewards_adj['subjective_prob'] = min_rewards_adj['certain_reward'] / min_rewards_adj["risky_reward"]

# Adjust用の図を新規作成（上書き防止）
plt.figure()

for persona in min_rewards_adj['persona'].unique():
    sub_df_adj = min_rewards_adj[min_rewards_adj['persona'] == persona].sort_values('p')

    x_adj = sub_df_adj['p'].values
    y_adj = sub_df_adj['subjective_prob'].values

    # LOWESSで滑らかな曲線を作成
    lowess_smoothed = lowess(y_adj, x_adj, frac=0.6)  # fracで滑らかさを調整 (0に近づけるほど元データ寄り、1に近づくほど滑らか)
    x_adj_smooth = lowess_smoothed[:, 0]
    y_adj_smooth = lowess_smoothed[:, 1]

    # 滑らかな曲線を描画
    plt.plot(x_adj_smooth, y_adj_smooth, label=persona)

    # 元の点も参考として描画
    plt.scatter(x_adj, y_adj, marker='o', alpha=0.5)

plt.plot([0, 1], [0, 1], 'k--', label='Y = X (Reference)')
plt.xlabel('Objective Probability')
plt.ylabel('Subjective Probability')
plt.title('Subjective-Objective Probability by Persona')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(title='Persona')
plt.grid(True)
plt.tight_layout()

plt.savefig(output_figure_adjust_path, dpi=300)
plt.close()

########## Invert Figure (50%以下の最大) ##########
# 条件を満たす最小certain_reward抽出
filtered_inv = df_csv[df_csv['prob_certain'] <= 0.5].copy()
if args.prompt.startswith("e"):
    max_rewards = filtered_inv.groupby(['persona', 'p', "risky_reward"])['certain_reward'].max().reset_index()
    max_rewards["certain_reward"] = max_rewards["certain_reward"].fillna(0)
elif args.prompt.startswith(("s", "c")):
    max_rewards = filtered_inv.groupby(['persona', 'p', "risky_reward"])['certain_reward'].min().reset_index()
    max_rewards["certain_reward"] = max_rewards["certain_reward"].fillna(max_ce)
max_rewards['subjective_prob'] = max_rewards['certain_reward'] / max_rewards["risky_reward"]

plt.figure()

for persona in max_rewards['persona'].unique():
    sub_df_inv = max_rewards[max_rewards['persona'] == persona].sort_values('p')

    x_inv = sub_df_inv['p'].values
    y_inv = sub_df_inv['subjective_prob'].values

    # LOWESSで滑らかな曲線を作成
    lowess_smoothed = lowess(y_inv, x_inv, frac=0.6)  # fracで滑らかさを調整 (0に近づけるほど元データ寄り、1に近づくほど滑らか)
    x_inv_smooth = lowess_smoothed[:, 0]
    y_inv_smooth = lowess_smoothed[:, 1]

    # 滑らかな曲線を描画
    plt.plot(x_inv_smooth, y_inv_smooth, label=persona)

    # 元の点も参考として描画
    plt.scatter(x_inv, y_inv, marker='o', alpha=0.5)

plt.plot([0, 1], [0, 1], 'k--', label='Y = X (Reference)')
plt.xlabel('Objective Probability')
plt.ylabel('Subjective Probability')
plt.title('Subjective-Objective Probability by Persona')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(title='Persona')
plt.grid(True)
plt.tight_layout()

plt.savefig(output_figure_invert_path, dpi=300)
plt.close()

############ Adjusted Median Figure ##########
# 条件を満たす最小certain_reward抽出
filtered_min_median = df_csv[df_csv['prob_certain_adjust'] >= 0.5].copy()
filtered_max_median = df_csv[df_csv['prob_certain_adjust'] <= 0.5].copy()
if args.prompt.startswith("e"):
    median_rewards = filtered_min_median.groupby(['persona', 'p', "risky_reward"])['certain_reward'].min().reset_index()
    median_rewards.rename(columns={'certain_reward': 'certain_reward_min'}, inplace=True)
    max_rewards_median = filtered_max_median.groupby(['persona', 'p', "risky_reward"])['certain_reward'].max().reset_index()
    max_rewards_median.rename(columns={'certain_reward': 'certain_reward_max'}, inplace=True)
    median_rewards = median_rewards.merge(max_rewards_median, on=['persona', 'p', 'risky_reward'], how='left')
    median_rewards['certain_reward_min'] = median_rewards['certain_reward_min'].fillna(max_ce)
    median_rewards['certain_reward_max'] = median_rewards['certain_reward_max'].fillna(0)
elif args.prompt.startswith(("s", "c")):
    median_rewards = filtered_min_median.groupby(['persona', 'p', "risky_reward"])['certain_reward'].max().reset_index()
    median_rewards.rename(columns={'certain_reward': 'certain_reward_min'}, inplace=True)
    max_rewards_median = filtered_max_median.groupby(['persona', 'p', "risky_reward"])['certain_reward'].min().reset_index()
    max_rewards_median.rename(columns={'certain_reward': 'certain_reward_max'}, inplace=True)
    median_rewards = median_rewards.merge(max_rewards_median, on=['persona', 'p', 'risky_reward'], how='left')
    median_rewards['certain_reward_min'] = median_rewards['certain_reward_min'].fillna(0)
    median_rewards['certain_reward_max'] = median_rewards['certain_reward_max'].fillna(max_ce)
median_rewards['certain_reward_median'] = (median_rewards['certain_reward_min'] + median_rewards['certain_reward_max']) / 2
median_rewards['subjective_prob'] = median_rewards['certain_reward_median'] / median_rewards["risky_reward"]

plt.figure()

for persona in median_rewards['persona'].unique():
    sub_df_median = median_rewards[median_rewards['persona'] == persona].sort_values('p')

    x_median = sub_df_median['p'].values
    y_median = sub_df_median['subjective_prob'].values

    # LOWESSで滑らかな曲線を作成
    lowess_smoothed = lowess(y_median, x_median, frac=0.6)  # fracで滑らかさを調整 (0に近づけるほど元データ寄り、1に近づくほど滑らか)
    x_median_smooth = lowess_smoothed[:, 0]
    y_median_smooth = lowess_smoothed[:, 1]

    # 滑らかな曲線を描画
    plt.plot(x_median_smooth, y_median_smooth, label=persona)

    # 元の点も参考として描画
    plt.scatter(x_median, y_median, marker='o', alpha=0.5)

plt.plot([0, 1], [0, 1], 'k--', label='Y = X (Reference)')
plt.xlabel('Objective Probability')
plt.ylabel('Subjective Probability')
plt.title('Subjective-Objective Probability by Persona')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(title='Persona')
plt.grid(True)
plt.tight_layout()

plt.savefig(output_figure_median_path, dpi=300)
plt.close()
