"""二值血管掩膜的三维渲染。

用 marching cubes 取等值面 + matplotlib 的 Poly3DCollection 上色，
而不是 3D Slicer 手工渲染 —— 定性对比图**四张必须完全同视角同缩放**，
程序化渲染里这是一个参数的事，手工对齐反而最容易出错。

网格很小（step_size=2 时约 2 万面，0.1 秒），不需要额外的加速手段。
"""

import numpy as np
from scipy import ndimage
from skimage import measure
from matplotlib.colors import LightSource, to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _mesh(mask, step=2):
    if mask.sum() == 0:
        return None
    verts, faces, _, _ = measure.marching_cubes(mask.astype(np.uint8), 0.5,
                                                step_size=step)
    return verts, faces


def _shaded(verts, faces, colour, azdeg=225, altdeg=55):
    """按面法线做朗伯光照，纯色会让血管糊成一团剪影。"""
    tris = verts[faces]
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(norm == 0, 1, norm)
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    d = ls.direction
    shade = np.clip(n @ d, 0, 1)
    base = np.array(to_rgb(colour))
    # 0.45 环境光 + 0.55 漫反射，避免背光面全黑
    rgb = base[None, :] * (0.45 + 0.55 * shade[:, None])
    return np.clip(rgb, 0, 1)


def add_mask(ax, mask, colour, step=2, alpha=1.0, zorder=1, rasterized=True):
    m = _mesh(mask, step)
    if m is None:
        return
    verts, faces = m
    coll = Poly3DCollection(verts[faces], linewidths=0)
    # 两万多个面若以矢量存入 PDF，单张图就有 4MB；光栅化后约 200KB，
    # 而文字与坐标轴仍是矢量。savefig.dpi 在 style.py 里设为 450。
    coll.set_rasterized(rasterized)
    coll.set_facecolor(_shaded(verts, faces, colour))
    coll.set_alpha(alpha)
    coll.set_zorder(zorder)
    ax.add_collection3d(coll)


def split_components(mask, keep=2):
    """拆成 (主体, 多余碎块)。主体 = 最大的 keep 个连通域。

    金标准是左右两棵冠脉树，所以 keep=2；预测里多出来的连通域正是
    Betti-0 误差的来源，单独上色就能让「碎成 11 块」一眼可见。
    """
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask, np.zeros_like(mask)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    order = np.argsort(sizes)[::-1] + 1
    main_ids = set(order[:keep].tolist())
    main = np.isin(lab, list(main_ids))
    return main, mask & ~main


def bbox_frame(mask, pad=6):
    """返回 (centre, half)：包住前景的立方体框。

    定性对比的四张图必须用**同一个**框（通常取金标准的），否则各自贴合
    自己的前景，缩放就不一样了，视觉上不可比。
    """
    idx = np.argwhere(mask)
    lo, hi = idx.min(axis=0) - pad, idx.max(axis=0) + pad
    centre = (lo + hi) / 2.0
    half = float((hi - lo).max()) / 2.0
    return centre, half


def frame(ax, shape, elev, azim, zoom=1.0, centre=None, half=None):
    """统一相机与坐标范围 —— 各 panel 必须调用同样的参数才可比。"""
    ax.view_init(elev=elev, azim=azim)
    c = np.array(shape) / 2 if centre is None else np.asarray(centre, float)
    half = (max(shape) / 2 if half is None else half) / zoom
    ax.set_xlim(c[0] - half, c[0] + half)
    ax.set_ylim(c[1] - half, c[1] + half)
    ax.set_zlim(c[2] - half, c[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_proj_type("ortho")
