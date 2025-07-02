import argparse
import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
#from io import StringIO


# ===== 引数の設定 =====
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model alias (e.g., llama3, deepseek)")
args = parser.parse_args()

# パスの設定
input_path =f"outputs/extract_data/{args.model}_extracted.csv"
output_csv_path = f"outputs/evaluate/{args.model}_evalusted.csv"
output_figure_path = f"outputs/figures/cefigure_{args.model}.png"
output_figure_adjust_path = f"outputs/figures/cefigure_{args.model}_adjust.png"
output_figure_invert_path = f"outputs/figures/cefigure_{args.model}_invert.png"
output_figure_median_path = f"outputs/figures/cefigure_{args.model}_median.png"       

# CSVをDataFrameとして読み込み
df = pd.read_csv(input_path, encoding="utf-8-sig")

# それぞれの変数を格納する
total_numb = len(df)
answer_choice_counts = df["answer_choice"].value_counts().to_dict()
count_answer_A = answer_choice_counts.get("Option A", 0)
count_answer_B = answer_choice_counts.get("Option B", 0)

combo_counts = (
    df.groupby(["answer_choice", "answer_meaning"])
    .size()
    .reset_index(name="count")
)
# combo_countsはDataFrame。必要ならdictに変換も可:
combo_dict = {
    (row["answer_choice"], row["answer_meaning"]): row["count"]
    for _, row in combo_counts.iterrows()
}

detailed_counts = (
    df.groupby(["persona", "p", "certain_reward", "risky_reward", "answer_choice", "answer_meaning"])
    .size()
    .unstack(["answer_choice", "answer_meaning"], fill_value=0)
    .reset_index()
)

# カラム名をフラットにする
detailed_counts.columns.name = None
detailed_counts.columns = ['persona', 'p', 'certain_reward', 'risky_reward',
                           'count_A_certain', 'count_A_risky', 'count_B_certain', 'count_B_risky']

#  total_numb列を作成
detailed_counts["total_numb"] = (
    detailed_counts["count_A_certain"] +
    detailed_counts["count_A_risky"] +
    detailed_counts["count_B_certain"] +
    detailed_counts["count_B_risky"]
)

# total_numb列を risky_reward の後ろに移動
cols = detailed_counts.columns.tolist()
# 現在のインデックスを取得し、total_numb を挿入
risky_index = cols.index("risky_reward")
# 一旦 total_numb を除去し、挿入位置に再挿入
cols.remove("total_numb")
cols.insert(risky_index + 1, "total_numb")
detailed_counts = detailed_counts[cols]
detailed_counts["count_risky"] = detailed_counts["count_A_risky"] + detailed_counts["count_B_risky"]
detailed_counts["count_certain"] = detailed_counts["count_A_certain"] + detailed_counts["count_B_certain"]
detailed_counts["prob_certain"] = (detailed_counts["count_certain"] 
                                   / (detailed_counts["count_certain"] 
                                      + detailed_counts["count_risky"])).round(3)
detailed_counts["count_risky_adjust"] = ((detailed_counts["count_A_risky"] 
                                          * total_numb / count_answer_A)
                                         + (detailed_counts["count_B_risky"] 
                                            * total_numb / count_answer_B)).round(3)
detailed_counts["count_certain_adjust"] = ((detailed_counts["count_A_certain"] 
                                            * total_numb / count_answer_A) 
                                           + (detailed_counts["count_B_certain"] 
                                              * total_numb / count_answer_B)).round(3)
detailed_counts["prob_certain_adjust"] = (detailed_counts["count_certain_adjust"]
                                          / (detailed_counts["count_certain_adjust"] 
                                             + detailed_counts["count_risky_adjust"])).round(3)

# 必要なら表示
detailed_counts.to_csv(output_csv_path, index=False)



# Figure作成
# 読み込みと加工
df_csv = pd.read_csv(output_csv_path)

# 条件を満たす最小certain_reward抽出
filtered = df_csv[df_csv['prob_certain'] >= 0.5]
min_rewards = filtered.groupby(['persona', 'p', "risky_reward"])['certain_reward'].min().reset_index()
min_rewards['certain_reward'] = min_rewards['certain_reward'].fillna(1000)  # NaNを1000で埋める
min_rewards['subjective_prob'] = min_rewards['certain_reward'] / min_rewards["risky_reward"]

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


########## Adjusted Figure ##########
# 条件を満たす最小certain_reward抽出
filtered = df_csv[df_csv['prob_certain_adjust'] >= 0.5]
min_rewards = filtered.groupby(['persona', 'p', "risky_reward"])['certain_reward'].min().reset_index()
min_rewards['certain_reward'] = min_rewards['certain_reward'].fillna(1000)  # NaNを1000で埋める
min_rewards['subjective_prob'] = min_rewards['certain_reward'] / min_rewards["risky_reward"]

# Adjust用の図を新規作成（上書き防止）
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

plt.savefig(output_figure_adjust_path, dpi=300)

########## Invert Figure (50%以下の最大) ##########
# 条件を満たす最小certain_reward抽出
filtered = df_csv[df_csv['prob_certain'] <= 0.5]
max_rewards = filtered.groupby(['persona', 'p', "risky_reward"])['certain_reward'].max().reset_index()
max_rewards["certain_reward"] = max_rewards["certain_reward"].fillna(0)  # NaNを0で埋める
max_rewards['subjective_prob'] = max_rewards['certain_reward'] / max_rewards["risky_reward"]

plt.figure()

for persona in max_rewards['persona'].unique():
    sub_df = max_rewards[max_rewards['persona'] == persona].sort_values('p')

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

plt.savefig(output_figure_invert_path, dpi=300)


############ Median Figure ##########
# 条件を満たす最小certain_reward抽出
filtered_min = df_csv[df_csv['prob_certain'] >= 0.5]
filtered_max = df_csv[df_csv['prob_certain'] <= 0.5]
median_rewards = filtered_min.groupby(['persona', 'p', "risky_reward"])['certain_reward'].min().reset_index()
median_rewards.rename(columns={'certain_reward': 'certain_reward_min'}, inplace=True)
max_rewards = filtered_max.groupby(['persona', 'p', "risky_reward"])['certain_reward'].max().reset_index()
max_rewards.rename(columns={'certain_reward': 'certain_reward_max'}, inplace=True)
median_rewards = median_rewards.merge(max_rewards, on=['persona', 'p', 'risky_reward'], how='left')
median_rewards['certain_reward_min'] = median_rewards['certain_reward_min'].fillna(1000)
median_rewards['certain_reward_max'] = median_rewards['certain_reward_max'].fillna(0)
median_rewards['certain_reward_median'] = (median_rewards['certain_reward_min'] + median_rewards['certain_reward_max']) / 2
median_rewards['subjective_prob'] = median_rewards['certain_reward_median'] / min_rewards["risky_reward"]

plt.figure()

for persona in median_rewards['persona'].unique():
    sub_df = median_rewards[median_rewards['persona'] == persona].sort_values('p')

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

plt.savefig(output_figure_median_path, dpi=300)