"""论文图表的数据读取层。

所有图只从 `.kb/results.md`「复现所需的确切配置」里点名的 csv 取数，
路径集中在这里，避免各图各拿一份、口径漂移 —— 本项目已因此发生过多次结论反转。
"""

import csv
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# key -> (相对路径, 该文件里代表「最终结果」的列前缀)
SOURCES = {
    # 单轴 + TTA：Stage-1 的对照
    "single_tta":  ("runs/exp_2p5d/test_final_tta.csv", "pp"),
    # 三正交 v2：Stage-1 基线。⚠️ 必须是 _v2，无后缀那份是 epoch=36 的次优权重
    "tri":         ("runs/exp_tri2p5d/test_metrics_tri_mean050_v2.csv", "pp"),
    # Stage-2 最优（无门控）。同一文件里 s1_* 是它的输入，即三正交
    "final":       ("runs/stage2_tri_nogate/test_metrics.csv", "s2"),
    "final_input": ("runs/stage2_tri_nogate/test_metrics.csv", "s1"),
    # Stage-2 消融
    "abl_full":    ("runs/stage2_tri/test_metrics_stage2_tri.csv", "s2"),
    "abl_nocldice": ("runs/stage2_tri_nocldice/test_metrics.csv", "s2"),
}


class MissingCsv(FileNotFoundError):
    pass


def path_of(key):
    rel, _ = SOURCES[key]
    return os.path.join(REPO, rel)


def load(key, metric):
    """返回 {case_id: value}。metric 用无前缀名，如 'dice' / 'hd95' / 'n_pred'。"""
    rel, prefix = SOURCES[key]
    p = os.path.join(REPO, rel)
    if not os.path.exists(p):
        raise MissingCsv(
            f"缺少 {rel}\n"
            f"→ 在本地仓库根目录跑：./scripts/fetch_paper_csv.sh")
    col = f"{prefix}_{metric}"
    out = {}
    with open(p) as fh:
        for r in csv.DictReader(fh):
            if col not in r:
                raise KeyError(f"{rel} 里没有列 {col}，实际列：{list(r)[:8]}…")
            out[r["id"]] = float(r[col])
    return out


def paired(key_a, key_b, metric):
    """按 case id 配对，返回 (ids, a_values, b_values)，顺序一致。"""
    a, b = load(key_a, metric), load(key_b, metric)
    ids = sorted(set(a) & set(b), key=int)
    return ids, [a[i] for i in ids], [b[i] for i in ids]
