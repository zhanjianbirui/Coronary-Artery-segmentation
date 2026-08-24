#!/usr/bin/env python3
"""
src/spatial_prior.py — 基于空间先验的连通分量过滤
==================================================================
动机（来自 2026-08-24 的误差诊断，见 .kb/logs/2026-08-24.md）：

  测试集 HD95 是重尾分布（中位数 14.91，均值 21.20），12 例 HD95>50
  贡献了均值的 23%。逐例相关性显示：

      corr(HD95, precision) = -0.665
      corr(HD95, recall)    = -0.229

  HD95>50 那组的 recall 和好病例几乎一样（0.79 vs 0.82），precision 却
  低了 0.14。也就是说 —— **失败模式不是漏检，是多画**：模型在个别病例
  上把主动脉/静脉/其他管状结构认成了冠脉，形成一块**体积很大、但离真实
  冠脉树很远**的假阳。

  现有后处理只有体积阈值（remove_small_components），这块假阳因为够大
  所以删不掉，而它离冠脉树远，HD95 直接爆到 80~126。

本模块补上后处理链里缺失的**空间/解剖先验**：冠脉是一棵（或两棵）连续的
树，真正的血管分支必然长在主干附近；离主干几十毫米之外还孤零零飘着的
一大块，解剖上不可能是冠脉。

算法：
  1. 按体积排序，取 top-N 个最大分量作为"锚"（主干）
  2. 其余分量：算到锚的最小距离（**毫米**，不是体素）
  3. 距离 <= max_dist_mm 的保留，否则删除
  4. （可选）链式生长：刚保留的分量本身也成为锚，再扫一轮，
     这样"主干 → 中间碎片 → 远端分支"的链条不会被中间断开而误删

为什么锚是 top-N 而不是"最大的那一个"：
  测试集真值的连通分量数分布是 {1:3, 2:85, 3:55, 4:32, 5:12, ...}，
  85/200 例的真值本身就有 2 个分量 —— 左冠和右冠是两棵**互不相连**的树。
  只保留最大分量会把整条右冠直接删掉。默认 n_anchor=2 正对应这个解剖事实。

用法（作为模块被 predict.py / predict_tri.py 调用），也可独立自测：
  python src/spatial_prior.py
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

try:
    from skimage.morphology import skeletonize
    _HAS_SKEL = True
except Exception:                                   # pragma: no cover
    _HAS_SKEL = False


def _component_scores(mask, anchor_by="skeleton"):
    """标记连通分量并给每个分量打分，返回 (lab, 按分数降序的标签, 分数, 体积)。

    **打分方式是这个模块的成败关键。**

    用体积排序是错的：假阳往往是主动脉/静脉那样**紧实的大团块**，体积可以
    轻松超过整棵冠脉树（合成测试里 4320 vs ~1000 体素）。那样团块会被选成
    "主干"，真正的血管树反而被当成远处杂物删掉 —— 完全适得其反。

    冠脉的判别特征不是"大"，而是"**长**"。所以默认按**骨架长度**（中心线
    体素数）排序：一棵冠脉树的中心线很长，一个紧实团块的骨架很短。
    这也正好和我们真正在优化的 clDice 指标同源。
    """
    lab, n = ndimage.label(mask)
    if n == 0:
        return lab, [], {}, {}

    index = np.arange(1, n + 1)
    sizes = ndimage.sum(np.ones_like(lab, dtype=np.uint8), lab, index=index)
    size_of = {i + 1: int(sizes[i]) for i in range(n)}

    if anchor_by == "skeleton" and _HAS_SKEL:
        # 整块 mask 只骨架化一次，再按标签统计各分量的中心线长度
        skel = skeletonize(mask.astype(bool))
        lengths = ndimage.sum(skel, lab, index=index)
        score_of = {i + 1: float(lengths[i]) for i in range(n)}
        # 骨架化对极小分量可能给出 0，退化到体积避免并列 0 时排序随机
        if max(score_of.values(), default=0.0) <= 0:
            score_of = {k: float(v) for k, v in size_of.items()}
    else:
        score_of = {k: float(v) for k, v in size_of.items()}

    order = sorted(score_of, key=lambda i: score_of[i], reverse=True)
    return lab, order, score_of, size_of


def _coords_mm(lab, comp_id, spacing):
    """取某个分量的体素坐标，换算成毫米。"""
    return np.argwhere(lab == comp_id) * spacing


def _subsample(points, max_pts):
    """点数超过 max_pts 时等距抽稀。

    抽稀只会让算出的最小距离偏大（可能错过真正最近的点），
    也就是让过滤**更激进**一点，不会出现"该删的没删"。
    体素 0.5mm、抽稀步长个位数时，误差远小于 max_dist_mm 的量级。
    """
    if max_pts <= 0 or len(points) <= max_pts:
        return points
    stride = int(np.ceil(len(points) / max_pts))
    return points[::stride]


def spatial_prior_filter(mask, spacing=0.5, n_anchor=2, max_dist_mm=20.0,
                         anchor_min_frac=0.10, chain=True, max_iter=10,
                         max_anchor_pts=200_000, anchor_by="skeleton",
                         return_info=False):
    """
    用空间先验过滤连通分量：只保留主干及其附近的分量。

    参数
    ----
    mask : ndarray  二值三维预测（不会被修改，返回新数组）
    spacing : float 或 长度3的序列  体素物理尺寸（mm），预处理后是各向同性 0.5
    n_anchor : int  取前几"长"的分量作锚。默认 2 = 左冠 + 右冠
    max_dist_mm : float  分量到锚的最小距离阈值（mm），超过则删
    anchor_min_frac : float  第 2..N 个锚的分数至少要达到第一名的这个比例，
                             否则不升格为锚（避免只有一棵树时把噪声当第二棵）
    chain : bool  是否链式生长（保留下来的分量也成为锚，再扫一轮）
    max_iter : int  链式生长的最大轮数
    max_anchor_pts : int  锚点集抽稀上限，控制 KDTree 开销
    anchor_by : "skeleton" | "volume"  选锚依据，见 _component_scores。
                默认 skeleton —— 用 volume 会被大块假阳骗过去
    return_info : bool  为 True 时额外返回诊断信息 dict

    返回
    ----
    out : ndarray  过滤后的二值预测（dtype 与输入一致）
    info : dict    仅当 return_info=True。含 n_before/n_after/n_dropped/
                   dropped_voxels/max_dropped_dist_mm/anchor_ids
    """
    mask = np.asarray(mask)
    spacing = np.asarray(spacing, dtype=np.float64)
    if spacing.ndim == 0:
        spacing = np.repeat(spacing, 3)

    lab, order, score_of, size_of = _component_scores(mask, anchor_by)
    n_before = len(order)

    empty_info = {"n_before": n_before, "n_after": n_before, "n_dropped": 0,
                  "dropped_voxels": 0, "max_dropped_dist_mm": 0.0,
                  "anchor_ids": list(order)}
    # 0 或 1 个分量时没有"远处的孤块"可言，原样返回
    if n_before <= 1:
        out = mask.copy()
        return (out, empty_info) if return_info else out

    # ---- 选锚：top-N 分数最高的分量，且第 2..N 个要够高 ----
    top_score = score_of[order[0]]
    anchors = [order[0]]
    for cid in order[1:max(1, n_anchor)]:
        if score_of[cid] >= anchor_min_frac * top_score:
            anchors.append(cid)
    kept = set(anchors)
    candidates = [c for c in order if c not in kept]

    # ---- 距离判定（可链式生长）----
    dropped_dists = []
    anchor_pts = np.concatenate([_coords_mm(lab, c, spacing) for c in anchors])
    tree = cKDTree(_subsample(anchor_pts, max_anchor_pts))

    for _ in range(max_iter if chain else 1):
        newly_kept = []
        still_pending = []
        for cid in candidates:
            pts = _coords_mm(lab, cid, spacing)
            d = tree.query(pts, k=1)[0].min()
            if d <= max_dist_mm:
                newly_kept.append((cid, pts))
            else:
                still_pending.append((cid, d))
        for cid, _pts in newly_kept:
            kept.add(cid)
        candidates = [cid for cid, _d in still_pending]
        if not chain or not newly_kept or not candidates:
            dropped_dists = [d for _cid, d in still_pending]
            break
        # 新保留的分量升格为锚，重建 KDTree 后再扫一轮
        anchor_pts = np.concatenate(
            [anchor_pts] + [pts for _cid, pts in newly_kept])
        tree = cKDTree(_subsample(anchor_pts, max_anchor_pts))
        dropped_dists = [d for _cid, d in still_pending]

    out = np.isin(lab, list(kept)).astype(mask.dtype)

    if not return_info:
        return out
    dropped = [c for c in order if c not in kept]
    info = {
        "n_before": n_before,
        "n_after": len(kept),
        "n_dropped": len(dropped),
        "dropped_voxels": int(sum(size_of[c] for c in dropped)),
        "max_dropped_dist_mm": float(max(dropped_dists)) if dropped_dists else 0.0,
        "anchor_ids": anchors,
    }
    return out, info


def _selftest():
    """自测：三个场景，每个都对应一个真实存在的坑。"""
    spacing = 0.5
    ok = True

    def check(name, got, want=True):
        nonlocal ok
        ok &= (bool(got) == want)
        print(f"  [{'通过' if bool(got) == want else '失败'}] {name}")

    # ---- 场景 1：基本行为 ----
    vol = np.zeros((120, 120, 120), dtype=np.uint8)
    vol[60, 60, 10:80] = 1          # 左冠主干
    vol[20, 20, 10:70] = 1          # 右冠（独立的第二棵树）
    vol[60, 64, 40:60] = 1          # 近处分支：离主干 4 体素 = 2mm
    vol[110, 110, 40:70] = 1        # 远处孤块（假阳）
    n0 = ndimage.label(vol)[1]
    print(f"\n场景1 基本行为（过滤前 {n0} 个分量）")
    out, info = spatial_prior_filter(vol, spacing=spacing, n_anchor=2,
                                     max_dist_mm=10.0, return_info=True)
    check("远处孤块被删", not out[110, 110, 50])
    check("近处分支保留", out[60, 64, 50])
    check("右冠保留（n_anchor=2 的意义）", out[20, 20, 50])
    check("左冠主干保留", out[60, 60, 50])
    check("输入未被就地修改", ndimage.label(vol)[1] == n0)
    print(f"  诊断: n_dropped={info['n_dropped']} "
          f"dropped_voxels={info['dropped_voxels']} "
          f"max_dropped_dist={info['max_dropped_dist_mm']:.1f}mm")

    # ---- 场景 2：n_anchor=1 会误删右冠 ----
    print("\n场景2 n_anchor 的必要性")
    out1 = spatial_prior_filter(vol, spacing=spacing, n_anchor=1,
                                max_dist_mm=10.0)
    check("n_anchor=1 时右冠被误删（所以默认取 2）", not out1[20, 20, 50])

    # ---- 场景 3：回归测试 —— 假阳团块比血管树体积更大 ----
    # 这是第一版按体积选锚时踩的坑：4320 体素的团块被当成主干，
    # 真正的冠脉树反而被删。必须靠骨架长度排序才能选对锚。
    print("\n场景3 回归：假阳团块体积 > 血管树（按体积选锚会翻车）")
    v2 = np.zeros((160, 160, 160), dtype=np.uint8)
    v2[80, 80, 20:120] = 1          # 左冠：细长
    v2[40, 40, 20:100] = 1          # 右冠：细长
    v2 = ndimage.binary_dilation(v2, iterations=1).astype(np.uint8)
    v2[140:152, 140:152, 60:90] = 1  # 紧实假阳团块，4320 体素
    tree_vox = int(v2.sum() - 4320)
    print(f"  团块 4320 体素 vs 两棵血管树合计 {tree_vox} 体素")

    out_skel = spatial_prior_filter(v2, spacing=spacing, n_anchor=2,
                                    max_dist_mm=20.0, anchor_by="skeleton")
    check("skeleton 选锚：团块被删", not out_skel[145, 145, 75])
    check("skeleton 选锚：左冠保留", out_skel[80, 80, 60])
    check("skeleton 选锚：右冠保留", out_skel[40, 40, 60])

    out_vol = spatial_prior_filter(v2, spacing=spacing, n_anchor=2,
                                   max_dist_mm=20.0, anchor_by="volume")
    check("volume 选锚：团块反被留下（说明 volume 不可用）",
          out_vol[145, 145, 75])

    print("\n" + ("全部通过 ✓" if ok else "有用例失败 ✗"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
