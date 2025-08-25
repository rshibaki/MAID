# import argparse
# import os
# import pandas as pd
# import numpy as np
# from statsmodels.nonparametric.smoothers_lowess import lowess
# import matplotlib.pyplot as plt
# #from io import StringIO

# PROMPTS_CONFIGS = {
#     # efficacy
#     "e1": "E1_ColorectalCan_PFS",
#     "e2": "E2_LiverCan_size",
#     "e3": "E3_HT_BP",
#     "e4": "E4_BA_peakflow",
#     "e5": "E5_Alzheimers_NPI",
#     # safety
#     "s1": "S1_PancreaticCan_anorexia",
#     "s2": "S2_SLE_infection",
#     "s3": "S3_PD_dyskinesia",
#     "s4": "S4_AF_bleeding",
#     "s5": "S5_Influenza_fever",
#     # cost
#     "c1": "C1_COVID19_hospitalization",
#     "c2": "C2_DM_hospitalization",
#     "c3": "C3_IPF_transplant",
#     "c6": "C6_Cancer_aiding",
#     "c7": "C7_HCV_aiding",
#     "c8": "C8_HT_medcost",
# }

# def story_class(story: str) -> str:
#     """先頭文字から E / S / C を返す"""
#     if not isinstance(story, str) or not story:
#         return "?"
#     head = story[0].upper()
#     return head if head in ("E", "S", "C") else "?"


# def ensure_dir(p: str):
#     os.makedirs(p, exist_ok=True)

# # ===== 引数の設定 =====
# parser = argparse.ArgumentParser()
# parser.add_argument("--model", type=str, required=True, help="Model alias (e.g., llama3, deepseek)")
# parser.add_argument("--prompt", type=str, choices=PROMPTS_CONFIGS.keys(), required=True, help="Prompt alias (e.g., e1")
# parser.add_argument("--d", type=str, default="d", help="demo or test")
# args = parser.parse_args()

# select_prompt = PROMPTS_CONFIGS[args.prompt]
# dt_judge = args.d

# # パスの設定
# if dt_judge == "d":
#     input_path =f"outputs/extract_data/demo/{args.model}/{args.model}_extracted_{select_prompt}.csv"
#     output_csv_path = f"outputs/evaluate/demo/{args.model}/{args.model}_evaluated_{select_prompt}.csv"
#     output_figure_path = f"outputs/figures/demo/{args.model}/{args.model}_sop_{select_prompt}.png"
#     output_figure_adjust_path = f"outputs/figures/demo/{args.model}/{args.model}_sop_{select_prompt}_adjust.png"
#     output_figure_invert_path = f"outputs/figures/demo/{args.model}/{args.model}_sop_{select_prompt}_invert.png"
#     output_figure_median_path = f"outputs/figures/demo/{args.model}/{args.model}_sop_{select_prompt}_median.png"
# elif args.d == "t":
#     input_path =f"outputs/extract_data/{args.model}/{args.model}_extracted_{select_prompt}.csv"
#     output_csv_path = f"outputs/evaluate/{args.model}/{args.model}_evaluated_{select_prompt}.csv"
#     output_figure_path = f"outputs/figures/{args.model}/{args.model}_sop_{select_prompt}.png"
#     output_figure_adjust_path = f"outputs/figures/{args.model}/{args.model}_sop_{select_prompt}_adjust.png"
#     output_figure_invert_path = f"outputs/figures/{args.model}/{args.model}_sop_{select_prompt}_invert.png"
#     output_figure_median_path = f"outputs/figures/{args.model}/{args.model}_sop_{select_prompt}_median.png"

# # CSVをDataFrameとして読み込み
# df = pd.read_csv(input_path, encoding="utf-8-sig")

# # # それぞれの変数を格納する
# total_numb = len(df)
# answer_choice_counts = df["answer_choice"].value_counts().to_dict()
# count_answer_A = answer_choice_counts.get("Option A", 0)
# count_answer_B = answer_choice_counts.get("Option B", 0)

# # ===== 1️⃣ persona,p,certain_reward,risky_reward,total_numb のベーステーブル作成 =====
# base = (
#     df.groupby(["persona", "p", "certain_reward", "risky_reward"])
#     .size()
#     .reset_index(name="total_numb")
# )

# # ===== 2️⃣ answer_choice × answer_meaning のクロス集計 =====
# counts = (
#     df.groupby(["persona", "p", "certain_reward", "risky_reward", "answer_choice", "answer_meaning"])
#     .size()
#     .reset_index(name="count")
# )

# # ★ 修正1: pivot 後に「期待する4列」を reindex で必ず作る
# expected_cols = pd.MultiIndex.from_product(
#     [["Option A", "Option B"], ["certain", "risky"]],
#     names=["answer_choice", "answer_meaning"],
# )

# pivot_counts = (
#     counts.pivot_table(
#         index=["persona", "p", "certain_reward", "risky_reward"],
#         columns=["answer_choice", "answer_meaning"],
#         values="count",
#         fill_value=0,
#         aggfunc="sum",
#     )
#     .reindex(columns=expected_cols, fill_value=0)  # ← ここがポイント
#     .reset_index()
# )

# # カラム名整理（Multiindex→Singleへ）
# pivot_counts.columns = [
#     "persona",
#     "p",
#     "certain_reward",
#     "risky_reward",
#     "count_A_certain",
#     "count_A_risky",
#     "count_B_certain",
#     "count_B_risky",
# ]

# # ===== 3️⃣ base とマージして detailed_counts 作成 =====
# detailed_counts = base.merge(
#     pivot_counts,
#     on=["persona", "p", "certain_reward", "risky_reward"],
#     how="left",
# ).fillna(0)

# # ===== 4️⃣ 集計列を追加 =====
# detailed_counts["count_risky"] = detailed_counts["count_A_risky"] + detailed_counts["count_B_risky"]
# detailed_counts["count_certain"] = detailed_counts["count_A_certain"] + detailed_counts["count_B_certain"]

# detailed_counts["prob_certain"] = (
#     detailed_counts["count_certain"] / (detailed_counts["count_certain"] + detailed_counts["count_risky"])
# ).round(3)

# detailed_counts["count_risky_adjust"] = (
#     (detailed_counts["count_A_risky"] * total_numb / count_answer_A)
#     + (detailed_counts["count_B_risky"] * total_numb / count_answer_B)
# ).round(3)
# detailed_counts["count_certain_adjust"] = (
#     (detailed_counts["count_A_certain"] * total_numb / count_answer_A)
#     + (detailed_counts["count_B_certain"] * total_numb / count_answer_B)
# ).round(3)
# detailed_counts["prob_certain_adjust"] = (
#     detailed_counts["count_certain_adjust"]
#     / (detailed_counts["count_certain_adjust"] + detailed_counts["count_risky_adjust"])).round(3)

# int_cols = [
#     "total_numb",
#     "count_A_certain",
#     "count_A_risky",
#     "count_B_certain",
#     "count_B_risky",
#     "count_risky",
#     "count_certain",
# ]
# detailed_counts[int_cols] = detailed_counts[int_cols].astype(int)

# # 必要なら表示
# detailed_counts.to_csv(output_csv_path, index=False)

# # ===== CSV保存後に総数を表示 =====
# cols_to_sum = [
#     "total_numb",
#     "count_A_certain",
#     "count_A_risky",
#     "count_B_certain",
#     "count_B_risky",
#     "count_risky",
#     "count_certain",
# ]
# print(f"count_answer_A: {count_answer_A}, count_answer_B: {count_answer_B}")

# # 合計を計算
# totals = detailed_counts[cols_to_sum].sum()

# print("===== 合計値 =====")
# for col, val in totals.items():
#     print(f"{col}: {val}")


# # Figure作成
# # 読み込みと加工
# df_csv = pd.read_csv(output_csv_path)
# max_ce = df.iloc[0]["risky_reward"]

# # 条件を満たす最小certain_reward抽出
# filtered = df_csv[df_csv['prob_certain'] >= 0.5].copy()
# if args.prompt.startswith("e"):
#     min_rewards = filtered.groupby(['persona', 'p', "risky_reward"])['certain_reward'].min ().reset_index()
#     min_rewards['certain_reward'] = min_rewards['certain_reward'].fillna(max_ce)
# elif args.prompt.startswith(("s", "c")):
#     min_rewards = filtered.groupby(['persona', 'p', "risky_reward"])['certain_reward'].max().reset_index()
#     min_rewards['certain_reward'] = min_rewards['certain_reward'].fillna(0)
# min_rewards['subjective_prob'] = min_rewards['certain_reward'] / min_rewards["risky_reward"]

# plt.figure()

# for persona in min_rewards['persona'].unique():
#     sub_df = min_rewards[min_rewards['persona'] == persona].sort_values('p')

#     x = sub_df['p'].values
#     y = sub_df['subjective_prob'].values

#     # LOWESSで滑らかな曲線を作成
#     lowess_smoothed = lowess(y, x, frac=0.6)  # fracで滑らかさを調整 (0に近づけるほど元データ寄り、1に近づくほど滑らか)
#     x_smooth = lowess_smoothed[:, 0]
#     y_smooth = lowess_smoothed[:, 1]

#     # 滑らかな曲線を描画
#     plt.plot(x_smooth, y_smooth, label=persona)

#     # 元の点も参考として描画
#     plt.scatter(x, y, marker='o', alpha=0.5)

# plt.plot([0, 1], [0, 1], 'k--', label='Y = X (Reference)')
# plt.xlabel('Objective Probability')
# plt.ylabel('Subjective Probability')
# plt.title('Subjective-Objective Probability by Persona')
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.legend(title='Persona')
# plt.grid(True)
# plt.tight_layout()

# plt.savefig(output_figure_path, dpi=300)
# plt.close()

# ########## Adjusted Figure ##########
# # 条件を満たす最小certain_reward抽出
# filtered_adj = df_csv[df_csv['prob_certain_adjust'] >= 0.5].copy()
# if args.prompt.startswith("e"):
#     min_rewards_adj = filtered_adj.groupby(['persona', 'p', "risky_reward"])['certain_reward'].min().reset_index()
#     min_rewards_adj['certain_reward'] = min_rewards_adj['certain_reward'].fillna(max_ce)
# elif args.prompt.startswith(("s", "c")):
#     min_rewards_adj = filtered_adj.groupby(['persona', 'p', "risky_reward"])['certain_reward'].max().reset_index()
#     min_rewards_adj['certain_reward'] = min_rewards_adj['certain_reward'].fillna(0)
# min_rewards_adj['subjective_prob'] = min_rewards_adj['certain_reward'] / min_rewards_adj["risky_reward"]

# # Adjust用の図を新規作成（上書き防止）
# plt.figure()

# for persona in min_rewards_adj['persona'].unique():
#     sub_df_adj = min_rewards_adj[min_rewards_adj['persona'] == persona].sort_values('p')

#     x_adj = sub_df_adj['p'].values
#     y_adj = sub_df_adj['subjective_prob'].values

#     # LOWESSで滑らかな曲線を作成
#     lowess_smoothed = lowess(y_adj, x_adj, frac=0.6)  # fracで滑らかさを調整 (0に近づけるほど元データ寄り、1に近づくほど滑らか)
#     x_adj_smooth = lowess_smoothed[:, 0]
#     y_adj_smooth = lowess_smoothed[:, 1]

#     # 滑らかな曲線を描画
#     plt.plot(x_adj_smooth, y_adj_smooth, label=persona)

#     # 元の点も参考として描画
#     plt.scatter(x_adj, y_adj, marker='o', alpha=0.5)

# plt.plot([0, 1], [0, 1], 'k--', label='Y = X (Reference)')
# plt.xlabel('Objective Probability')
# plt.ylabel('Subjective Probability')
# plt.title('Subjective-Objective Probability by Persona')
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.legend(title='Persona')
# plt.grid(True)
# plt.tight_layout()

# plt.savefig(output_figure_adjust_path, dpi=300)
# plt.close()

# ########## Invert Figure (50%以下の最大) ##########
# # 条件を満たす最小certain_reward抽出
# filtered_inv = df_csv[df_csv['prob_certain'] <= 0.5].copy()
# if args.prompt.startswith("e"):
#     max_rewards = filtered_inv.groupby(['persona', 'p', "risky_reward"])['certain_reward'].max().reset_index()
#     max_rewards["certain_reward"] = max_rewards["certain_reward"].fillna(0)
# elif args.prompt.startswith(("s", "c")):
#     max_rewards = filtered_inv.groupby(['persona', 'p', "risky_reward"])['certain_reward'].min().reset_index()
#     max_rewards["certain_reward"] = max_rewards["certain_reward"].fillna(max_ce)
# max_rewards['subjective_prob'] = max_rewards['certain_reward'] / max_rewards["risky_reward"]

# plt.figure()

# for persona in max_rewards['persona'].unique():
#     sub_df_inv = max_rewards[max_rewards['persona'] == persona].sort_values('p')

#     x_inv = sub_df_inv['p'].values
#     y_inv = sub_df_inv['subjective_prob'].values

#     # LOWESSで滑らかな曲線を作成
#     lowess_smoothed = lowess(y_inv, x_inv, frac=0.6)  # fracで滑らかさを調整 (0に近づけるほど元データ寄り、1に近づくほど滑らか)
#     x_inv_smooth = lowess_smoothed[:, 0]
#     y_inv_smooth = lowess_smoothed[:, 1]

#     # 滑らかな曲線を描画
#     plt.plot(x_inv_smooth, y_inv_smooth, label=persona)

#     # 元の点も参考として描画
#     plt.scatter(x_inv, y_inv, marker='o', alpha=0.5)

# plt.plot([0, 1], [0, 1], 'k--', label='Y = X (Reference)')
# plt.xlabel('Objective Probability')
# plt.ylabel('Subjective Probability')
# plt.title('Subjective-Objective Probability by Persona')
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.legend(title='Persona')
# plt.grid(True)
# plt.tight_layout()

# plt.savefig(output_figure_invert_path, dpi=300)
# plt.close()

# ############ Adjusted Median Figure ##########
# # 条件を満たす最小certain_reward抽出
# filtered_min_median = df_csv[df_csv['prob_certain_adjust'] >= 0.5].copy()
# filtered_max_median = df_csv[df_csv['prob_certain_adjust'] <= 0.5].copy()
# if args.prompt.startswith("e"):
#     median_rewards = filtered_min_median.groupby(['persona', 'p', "risky_reward"])['certain_reward'].min().reset_index()
#     median_rewards.rename(columns={'certain_reward': 'certain_reward_min'}, inplace=True)
#     max_rewards_median = filtered_max_median.groupby(['persona', 'p', "risky_reward"])['certain_reward'].max().reset_index()
#     max_rewards_median.rename(columns={'certain_reward': 'certain_reward_max'}, inplace=True)
#     median_rewards = median_rewards.merge(max_rewards_median, on=['persona', 'p', 'risky_reward'], how='left')
#     median_rewards['certain_reward_min'] = median_rewards['certain_reward_min'].fillna(max_ce)
#     median_rewards['certain_reward_max'] = median_rewards['certain_reward_max'].fillna(0)
# elif args.prompt.startswith(("s", "c")):
#     median_rewards = filtered_min_median.groupby(['persona', 'p', "risky_reward"])['certain_reward'].max().reset_index()
#     median_rewards.rename(columns={'certain_reward': 'certain_reward_min'}, inplace=True)
#     max_rewards_median = filtered_max_median.groupby(['persona', 'p', "risky_reward"])['certain_reward'].min().reset_index()
#     max_rewards_median.rename(columns={'certain_reward': 'certain_reward_max'}, inplace=True)
#     median_rewards = median_rewards.merge(max_rewards_median, on=['persona', 'p', 'risky_reward'], how='left')
#     median_rewards['certain_reward_min'] = median_rewards['certain_reward_min'].fillna(0)
#     median_rewards['certain_reward_max'] = median_rewards['certain_reward_max'].fillna(max_ce)
# median_rewards['certain_reward_median'] = (median_rewards['certain_reward_min'] + median_rewards['certain_reward_max']) / 2
# median_rewards['subjective_prob'] = median_rewards['certain_reward_median'] / median_rewards["risky_reward"]

# plt.figure()

# for persona in median_rewards['persona'].unique():
#     sub_df_median = median_rewards[median_rewards['persona'] == persona].sort_values('p')

#     x_median = sub_df_median['p'].values
#     y_median = sub_df_median['subjective_prob'].values

#     # LOWESSで滑らかな曲線を作成
#     lowess_smoothed = lowess(y_median, x_median, frac=0.6)  # fracで滑らかさを調整 (0に近づけるほど元データ寄り、1に近づくほど滑らか)
#     x_median_smooth = lowess_smoothed[:, 0]
#     y_median_smooth = lowess_smoothed[:, 1]

#     # 滑らかな曲線を描画
#     plt.plot(x_median_smooth, y_median_smooth, label=persona)

#     # 元の点も参考として描画
#     plt.scatter(x_median, y_median, marker='o', alpha=0.5)

# plt.plot([0, 1], [0, 1], 'k--', label='Y = X (Reference)')
# plt.xlabel('Objective Probability')
# plt.ylabel('Subjective Probability')
# plt.title('Subjective-Objective Probability by Persona')
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.legend(title='Persona')
# plt.grid(True)
# plt.tight_layout()

# plt.savefig(output_figure_median_path, dpi=300)
# plt.close()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_median_agg.py

目的:
- evaluate_csv.py の「############ Adjusted Median Figure ##########」と同じ考え方で、
  まず各シナリオごとに adjusted（prob_certain_adjust を用いた中央値）曲線を計算。
- その後、全シナリオ（ALL）/ Eのみ / Sのみ / Cのみ で横断的に平均化して作図。
- persona 別と socio_demo 別の両方を出力。

前提:
- extract_raw.py により、各シナリオごとの CSV が以下のように出力済みであること。
  demo: outputs/extract_data/demo/<model>/<model>_extracted_<story>.csv
  test: outputs/extract_data/<model>/<model>_extracted_<story>.csv
- 列名は evaluate_csv.py / extract_raw.py と同等（story, persona, socio_demo, p,
  risky_reward, certain_reward, answer_choice, answer_meaning など）。

使い方の例:
  python3 evaluate_median_agg.py --model llama3 --d d
  python3 evaluate_median_agg.py --model qwen2.5 --d t --topk_sociodemo 10
"""

import argparse
import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess


# ====== 対象シナリオ名（evaluate_csv.py と同じ命名規則を想定） ======
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

def story_class(story: str) -> str:
    """先頭文字から E / S / C を返す"""
    if not isinstance(story, str) or not story:
        return "?"
    head = story[0].upper()
    return head if head in ("E", "S", "C") else "?"

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# ====== evaluate_csv.py の計算を 1シナリオ単位で再現（Adjusted Median） ======
def compute_detailed_counts_one_story(df: pd.DataFrame, facet: str) -> pd.DataFrame:
    """
    1シナリオ DataFrame に対して、evaluate_csv.py と同等の
    prob_certain / prob_certain_adjust を計算し、pivot後の詳細表を返す。
    facet は 'persona' または 'socio_demo'
    """
    total_numb = len(df)
    answer_choice_counts = df["answer_choice"].value_counts().to_dict()
    # evaluate_csv.py と違い、0除算回避のため 0 のときは 1 に置き換える（安全策）
    count_answer_A = answer_choice_counts.get("Option A", 0) or 1
    count_answer_B = answer_choice_counts.get("Option B", 0) or 1

    base = (
        df.groupby([facet, "p", "certain_reward", "risky_reward"])
          .size().reset_index(name="total_numb")
    )
    counts = (
        df.groupby([facet, "p", "certain_reward", "risky_reward",
                    "answer_choice", "answer_meaning"])
          .size().reset_index(name="count")
    )

    expected_cols = pd.MultiIndex.from_product(
        [["Option A", "Option B"], ["certain", "risky"]],
        names=["answer_choice", "answer_meaning"],
    )

    pivot_counts = (
        counts.pivot_table(
            index=[facet, "p", "certain_reward", "risky_reward"],
            columns=["answer_choice", "answer_meaning"],
            values="count",
            fill_value=0,
            aggfunc="sum",
        )
        .reindex(columns=expected_cols, fill_value=0)
        .reset_index()
    )

    pivot_counts.columns = [
        facet, "p", "certain_reward", "risky_reward",
        "count_A_certain", "count_A_risky",
        "count_B_certain", "count_B_risky",
    ]

    detailed = base.merge(
        pivot_counts,
        on=[facet, "p", "certain_reward", "risky_reward"],
        how="left",
    ).fillna(0)

    # prob_certain
    detailed["count_risky"] = detailed["count_A_risky"] + detailed["count_B_risky"]
    detailed["count_certain"] = detailed["count_A_certain"] + detailed["count_B_certain"]

    denom = (detailed["count_certain"] + detailed["count_risky"]).replace(0, np.nan)
    detailed["prob_certain"] = (detailed["count_certain"] / denom).round(3)

    # prob_certain_adjust
    detailed["count_risky_adjust"] = (
        (detailed["count_A_risky"]   * total_numb / count_answer_A) +
        (detailed["count_B_risky"]   * total_numb / count_answer_B)
    ).round(3)
    detailed["count_certain_adjust"] = (
        (detailed["count_A_certain"] * total_numb / count_answer_A) +
        (detailed["count_B_certain"] * total_numb / count_answer_B)
    ).round(3)

    denom_adj = (detailed["count_certain_adjust"] + detailed["count_risky_adjust"]).replace(0, np.nan)
    detailed["prob_certain_adjust"] = (detailed["count_certain_adjust"] / denom_adj).round(3)

    return detailed


def adjusted_median_curve_one_story(detailed: pd.DataFrame, e_s_c: str, facet: str) -> pd.DataFrame:
    """
    evaluate_csv.py の「Adjusted Median Figure」を 1シナリオ分で再現し、
    [facet値, p, subjective_prob] の曲線点を返す。
    e_s_c: "E" or "S" or "C"
    """
    prob_col = "prob_certain_adjust"
    df = detailed.copy()

    above = df[df[prob_col] >= 0.5]
    below = df[df[prob_col] <= 0.5]

    if e_s_c == "E":
        lo = (above.groupby([facet, "p", "risky_reward"])["certain_reward"].min()
                    .rename("certain_min"))
        hi = (below.groupby([facet, "p", "risky_reward"])["certain_reward"].max()
                    .rename("certain_max"))
        picked = pd.concat([lo, hi], axis=1).reset_index()
        # evaluate_csv.py では max_ce を入れているが、各行の risky_reward で補完する方が安全
        picked["certain_min"] = picked["certain_min"].fillna(picked["risky_reward"])
        picked["certain_max"] = picked["certain_max"].fillna(0)
    else:  # S or C
        lo = (above.groupby([facet, "p", "risky_reward"])["certain_reward"].max()
                    .rename("certain_min"))
        hi = (below.groupby([facet, "p", "risky_reward"])["certain_reward"].min()
                    .rename("certain_max"))
        picked = pd.concat([lo, hi], axis=1).reset_index()
        picked["certain_min"] = picked["certain_min"].fillna(0)
        picked["certain_max"] = picked["certain_max"].fillna(picked["risky_reward"])

    picked["certain_reward_median"] = (picked["certain_min"] + picked["certain_max"]) / 2.0
    picked["subjective_prob"] = pd.to_numeric(picked["certain_reward_median"]) / pd.to_numeric(picked["risky_reward"])

    curve = picked[[facet, "p", "subjective_prob"]].copy()
    curve.rename(columns={facet: "series"}, inplace=True)  # 統一列名
    return curve


def plot_lowess(df_curve: pd.DataFrame, x_col: str, y_col: str,
                series_col: str, title: str, out_path: str):
    """LOWESS曲線 + 散布図で保存（色は指定しない）"""
    plt.figure()
    for key in sorted(df_curve[series_col].dropna().unique()):
        sub = df_curve[df_curve[series_col] == key].sort_values(x_col)
        if sub.empty:
            continue
        x = sub[x_col].values
        y = sub[y_col].values
        sm = lowess(y, x, frac=0.6)
        plt.plot(sm[:, 0], sm[:, 1], label=str(key))
        plt.scatter(x, y, marker='o', alpha=0.5)

    plt.plot([0, 1], [0, 1], 'k--', label='Y = X (Reference)')
    plt.xlabel('Objective Probability (p)')
    plt.ylabel('Subjective Probability')
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="例: llama3, qwen2.5 など")
    ap.add_argument("--d", default="d", choices=["d", "t"], help="demo/test")
    ap.add_argument("--outdir", default="outputs", help="出力ルート")
    ap.add_argument("--topk_sociodemo", type=int, default=12,
                    help="socio_demo の同時描画上限（系列が多すぎる場合の上位抽出）")
    args = ap.parse_args()

    # 入力CSVの探索パターン
    if args.d == "d":
        in_pattern = f"{args.outdir}/extract_data/demo/{args.model}/{args.model}_extracted_*.csv"
        fig_root = os.path.join(args.outdir, "figures_agg_median", "demo", args.model)
        csv_root = os.path.join(args.outdir, "evaluate_agg_median", "demo", args.model)
    else:
        in_pattern = f"{args.outdir}/extract_data/{args.model}/{args.model}_extracted_*.csv"
        fig_root = os.path.join(args.outdir, "figures_agg_median", args.model)
        csv_root = os.path.join(args.outdir, "evaluate_agg_median", args.model)

    paths = sorted(glob(in_pattern))
    if not paths:
        raise FileNotFoundError(f"No extracted CSVs found: {in_pattern}")

    ensure_dir(fig_root); ensure_dir(csv_root)

    # まず各シナリオで adjusted median 曲線を作成（persona / socio_demo）
    per_story_curves = []  # 列: story, class, facet_type, series, p, subjective_prob
    for pth in paths:
        df = pd.read_csv(pth, encoding="utf-8-sig")
        if "story" not in df.columns:
            # ファイル名から推定（_extracted_<story>.csv）
            story = os.path.splitext(os.path.basename(pth))[0].split("_extracted_")[-1]
            df["story"] = story
        story = str(df["story"].iloc[0])
        e_s_c = story_class(story)

        # persona
        det_persona = compute_detailed_counts_one_story(df, facet="persona")
        curve_p = adjusted_median_curve_one_story(det_persona, e_s_c=e_s_c, facet="persona")
        curve_p["story"] = story
        curve_p["class"] = e_s_c
        curve_p["facet_type"] = "persona"
        per_story_curves.append(curve_p)

        # socio_demo
        det_socio = compute_detailed_counts_one_story(df, facet="socio_demo")
        curve_s = adjusted_median_curve_one_story(det_socio, e_s_c=e_s_c, facet="socio_demo")
        curve_s["story"] = story
        curve_s["class"] = e_s_c
        curve_s["facet_type"] = "socio_demo"
        per_story_curves.append(curve_s)

    curves_all = pd.concat(per_story_curves, ignore_index=True)

    # バケット（ALL/E/S/C）×（persona/socio_demo）で横断平均
    buckets = {
        "ALL": None,
        "E": ["E"],
        "S": ["S"],
        "C": ["C"],
    }

    for bname, classes in buckets.items():
        if classes is None:
            bdf = curves_all.copy()
        else:
            bdf = curves_all[curves_all["class"].isin(classes)].copy()
        if bdf.empty:
            continue

        # persona
        persona_df = bdf[bdf["facet_type"] == "persona"]
        if not persona_df.empty:
            # p×series 単位でシナリオ平均
            avg_p = (persona_df.groupby(["series", "p"], as_index=False)["subjective_prob"].mean())
            # 保存（CSV + 図）
            csv_out = os.path.join(csv_root, f"{args.model}_curves_{bname}_by_persona_adjusted_median.csv")
            fig_out = os.path.join(fig_root, f"{args.model}_sop_{bname}_by_persona_adjusted_median.png")
            avg_p.to_csv(csv_out, index=False)
            plot_lowess(avg_p, "p", "subjective_prob", "series",
                        title=f"Adjusted Median (scenario-wise adjusted → aggregated) — {bname} — by Persona",
                        out_path=fig_out)

        # socio_demo（系列多数のため上位のみ描画）
        socio_df = bdf[bdf["facet_type"] == "socio_demo"]
        if not socio_df.empty:
            coverage = (socio_df.groupby("series")["p"].nunique().sort_values(ascending=False))
            top_series = list(coverage.head(max(1, args.topk_sociodemo)).index)
            avg_s = (socio_df[socio_df["series"].isin(top_series)]
                     .groupby(["series", "p"], as_index=False)["subjective_prob"].mean())
            csv_out2 = os.path.join(csv_root, f"{args.model}_curves_{bname}_by_sociodemo_adjusted_median.csv")
            fig_out2 = os.path.join(fig_root, f"{args.model}_sop_{bname}_by_sociodemo_adjusted_median.png")
            avg_s.to_csv(csv_out2, index=False)
            plot_lowess(avg_s, "p", "subjective_prob", "series",
                        title=f"Adjusted Median (scenario-wise adjusted → aggregated) — {bname} — by Socio-demo (top {args.topk_sociodemo})",
                        out_path=fig_out2)

    print("✅ Done. Figures:", fig_root)
    print("✅ Done. CSVs   :", csv_root)


if __name__ == "__main__":
    main()




