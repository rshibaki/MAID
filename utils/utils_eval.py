# utils_eval.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

def story_class(story: str) -> str:
    """先頭文字から E / S / C を返す"""
    if not isinstance(story, str) or not story:
        return "?"
    head = story[0].upper()
    return head if head in ("E", "S", "C") else "?"

def compute_detailed_counts(df: pd.DataFrame, facet: str) -> pd.DataFrame:
    """
    1シナリオ DataFrame に対して、evaluate_csv.py と同等の
    prob_certain / prob_certain_adjust を計算し、pivot後の詳細表を返す。
    facet は 'persona' または 'socio_demo'
    """
    assert facet in ("persona", "socio_demo"), f"facet must be persona or socio_demo, got {facet}"
    total_numb = len(df)
    answer_choice_counts = df["answer_choice"].value_counts().to_dict()

    # 0除算回避（回答が片側ゼロのケースでも落とさない）
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

def adjusted_median_curve(detailed: pd.DataFrame, e_s_c: str, facet: str) -> pd.DataFrame:
    """
    evaluate_csv.py の「Adjusted Median Figure」を再現。
    入力: compute_detailed_counts() の出力（単一シナリオ分）
    出力: 列 [facet値(=series), p, subjective_prob]
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
    curve.rename(columns={facet: "series"}, inplace=True)
    return curve

def lowess_plot(df_curve: pd.DataFrame, x_col: str, y_col: str,
                series_col: str, title: str, out_path: str) -> None:
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
