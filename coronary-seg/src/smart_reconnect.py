#!/usr/bin/env python3
"""
src/smart_reconnect.py — 方向感知的智能血管重连
==================================================================
替代之前"盲目连最近点"的简化版（那个是负优化）。
核心改进：只连接"顺着血管走向、且朝向彼此"的端点对。

算法：
  1. 提取骨架，找端点（骨架上邻居数==1的点）
  2. 对每个端点估计局部方向（端点往回追 L 个体素的走向）
  3. 配对判据（三个都满足才连）：
     a. 两端点距离 < max_gap
     b. 两端点方向大致相对（点积 < 阈值，即朝向彼此而非同向）
     c. 连线方向与两端点方向都一致（避免横向乱连）
  4. 用直线体素连接（可加粗保证连通）

用法（作为模块被 predict.py 调用），也可独立测试。
"""

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def _get_endpoints(skel):
    """找骨架端点：3D 邻域内除自己外只有1个骨架点。"""
    # 26邻域卷积核（中心为0）
    kernel = np.ones((3, 3, 3), dtype=int)
    kernel[1, 1, 1] = 0
    neighbor_count = ndimage.convolve(skel.astype(int), kernel,
                                      mode="constant")
    endpoints = np.argwhere((skel > 0) & (neighbor_count == 1))
    return endpoints


def _local_direction(skel_coords_set, endpoint, L=8):
    """
    估计端点的局部方向：从端点出发，沿骨架走 L 步，
    方向 = 归一化(端点 - L步后的点)，即"指向端点外侧"的方向。
    简化实现：在端点周围 L 半径内取骨架点，用 PCA 主方向，
    并让方向指向"离骨架质心更远"的一侧（即血管延伸方向）。
    """
    ex, ey, ez = endpoint
    # 取端点邻域内的骨架点
    nearby = []
    for (x, y, z) in skel_coords_set:
        if abs(x - ex) <= L and abs(y - ey) <= L and abs(z - ez) <= L:
            d2 = (x - ex) ** 2 + (y - ey) ** 2 + (z - ez) ** 2
            if d2 <= L * L:
                nearby.append((x, y, z))
    if len(nearby) < 3:
        return None
    pts = np.array(nearby, dtype=float)
    centroid = pts.mean(axis=0)
    # PCA 主方向
    cov = np.cov((pts - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_dir = eigvecs[:, -1]  # 最大特征值对应的方向
    # 让方向指向端点外侧：端点相对质心的方向
    out_vec = np.array(endpoint, dtype=float) - centroid
    if np.dot(main_dir, out_vec) < 0:
        main_dir = -main_dir
    norm = np.linalg.norm(main_dir)
    return main_dir / norm if norm > 0 else None


def smart_reconnect(mask, max_gap=20, L=8,
                    oppose_thr=-0.3, align_thr=0.5, thickness=1):
    """
    方向感知重连。
    参数：
      max_gap:     端点间最大连接距离（体素）
      L:           估计局部方向的邻域半径
      oppose_thr:  两端点方向点积 < 此值才算"朝向彼此"
                   （方向都指向外侧，相对时点积为负）
      align_thr:   连线方向与端点方向一致性阈值（越大越严格）
      thickness:   连接线加粗半径（1=细线）
    """
    mask = mask.astype(bool)
    skel = skeletonize(mask)
    endpoints = _get_endpoints(skel)
    if len(endpoints) < 2:
        return mask.astype(np.uint8)

    skel_set = set(map(tuple, np.argwhere(skel)))
    # 预计算每个端点的方向
    dirs = []
    for ep in endpoints:
        d = _local_direction(skel_set, ep, L=L)
        dirs.append(d)

    out = mask.copy()
    used = set()
    # 按距离从近到远尝试配对
    pairs = []
    for i in range(len(endpoints)):
        if dirs[i] is None:
            continue
        for j in range(i + 1, len(endpoints)):
            if dirs[j] is None:
                continue
            pi, pj = endpoints[i].astype(float), endpoints[j].astype(float)
            gap = np.linalg.norm(pi - pj)
            if gap < 1 or gap > max_gap:
                continue
            # 判据b: 两方向相对（都指外侧，相对时点积为负）
            if np.dot(dirs[i], dirs[j]) > oppose_thr:
                continue
            # 判据c: 连线方向与两端点方向一致
            link = (pj - pi) / (gap + 1e-6)   # i->j 方向
            # 端点i的方向应与link同向，端点j的方向应与-link同向
            if np.dot(dirs[i], link) < align_thr:
                continue
            if np.dot(dirs[j], -link) < align_thr:
                continue
            pairs.append((gap, i, j))

    # 按gap从小到大连接，每个端点只用一次
    pairs.sort()
    for gap, i, j in pairs:
        if i in used or j in used:
            continue
        used.add(i); used.add(j)
        p0 = endpoints[i].astype(float)
        p1 = endpoints[j].astype(float)
        steps = int(gap) + 1
        for t in np.linspace(0, 1, steps):
            pt = np.round(p0 + t * (p1 - p0)).astype(int)
            # 加粗
            for dx in range(-thickness + 1, thickness):
                for dy in range(-thickness + 1, thickness):
                    for dz in range(-thickness + 1, thickness):
                        x, y, z = pt[0] + dx, pt[1] + dy, pt[2] + dz
                        if (0 <= x < out.shape[0] and 0 <= y < out.shape[1]
                                and 0 <= z < out.shape[2]):
                            out[x, y, z] = True
    return out.astype(np.uint8)


if __name__ == "__main__":
    # 简单自测：造两段断开的直线血管，看能否连上
    vol = np.zeros((40, 40, 40), dtype=np.uint8)
    vol[20, 20, 5:18] = 1     # 第一段
    vol[20, 20, 23:35] = 1    # 第二段（中间断开 23-18=5 体素）
    print(f"重连前连通分量: {ndimage.label(vol)[1]}")
    out = smart_reconnect(vol, max_gap=20)
    print(f"重连后连通分量: {ndimage.label(out)[1]}")
    print(f"（应从 2 变成 1，说明断裂被正确连接）")