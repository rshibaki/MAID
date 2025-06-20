import argparse
import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
#from io import StringIO


# ===== 引数の設定 =====
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model alias (e.g., llama3, deepseek)")
parser.add_argument("--prompt", type=str, help="Which prompts to use")
args = parser.parse_args()

# パスの設定
input_path =f"new_test/outputs/extract_data/{args.model}_{args.prompt}_extracted.csv"
output_csv_path = f"new_test/outputs/evaluate/{args.model}_{args.prompt}_evalusted.csv"
output_figure_path = f"new_test/outputs/figures/cefigure_{args.model}_{args.prompt}.png"         

# CSVをDataFrameとして読み込み
df = pd.read_csv(input_path, encoding="utf-8-sig")

# "answer_mean" を数値に変換（certain: 1, risky: 0）
df["answer_numeric"] = df["answer_meaning"].map({"certain": 1, "risky": 0})

# 件数カウント用に risky = 0, certain = 1 としてそれぞれ集計
summary = (
    df.groupby(["persona", "p", "certain_reward", "risky_reward"])["answer_numeric"]
    .agg(
        prob_choose_certain="mean",  
        count_total="count",
        count_certain="sum"
    )
    .reset_index()
)

# risky の件数も追加（= 全体 - certain）
summary["count_risky"] = summary["count_total"] - summary["count_certain"]

# CSVとして保存
summary.to_csv(output_csv_path, index=False)

# Figure作成
# 読み込みと加工
df_csv = pd.read_csv(output_csv_path)

# 条件を満たす最小certain_reward抽出
filtered = df_csv[df_csv['prob_choose_certain'] >= 0.5]
min_rewards = filtered.groupby(['persona', 'p', "risky_reward"])['certain_reward'].min().reset_index()
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