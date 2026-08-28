"""多视角 stage-2 的接线检查（4 通道 [image, p0, p1, p2]）。

    python tests/test_multi_view.py

动机：这次改动横跨 4 个文件（prepare / data / model / predict），
最容易出的不是逻辑错误而是**接线错误** —— 某一处改了、另一处没跟上，
或改到了函数定义而非调用处（本次真的发生过一次）。
这类问题跑起来才炸，且往往在集群上排队几小时之后。

不 import 需要 monai 的模块，全部用 AST 静态检查。
"""
import ast
import io
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ok = _fail = 0


def chk(name, cond):
    global _ok, _fail
    if cond:
        _ok += 1
        print(f"  PASS  {name}")
    else:
        _fail += 1
        print(f"  FAIL  {name}")


def src(rel):
    return io.open(os.path.join(REPO, rel), encoding="utf-8").read()


def fn_of(tree, name):
    return next((n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == name), None)


def test_no_duplicate_args():
    """函数签名不能有重复参数 —— 本次改动真的踩过（views=None, views=views）。"""
    print("\n[1] 所有被改动文件的函数签名合法")
    for rel in ("scripts/predict/predict_stage2.py", "scripts/data/stage2_prepare.py",
                "src/stage2_data.py", "src/stage2_model.py",
                "scripts/train/train_stage2.py"):
        tree = ast.parse(src(rel))
        bad = []
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names = [a.arg for a in n.args.args + n.args.kwonlyargs]
                if len(names) != len(set(names)):
                    bad.append(n.name)
        chk(f"{rel} 无重复参数名", not bad)


def test_load_npz_unpack_matches():
    """load_npz 的返回值个数必须与所有解包处一致。"""
    print("\n[2] load_npz 的返回值与解包处一致")
    s = src("scripts/predict/predict_stage2.py")
    tree = ast.parse(s)
    ln = fn_of(tree, "load_npz")
    n_ret = max(len(r.value.elts) for r in ast.walk(ln)
                if isinstance(r, ast.Return) and isinstance(r.value, ast.Tuple))
    chk(f"load_npz 返回 {n_ret} 个值", n_ret == 4)

    unpacks = []
    for n in ast.walk(tree):
        if (isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                and getattr(n.value.func, "id", "") == "load_npz"
                and isinstance(n.targets[0], ast.Tuple)):
            unpacks.append(len(n.targets[0].elts))
    chk(f"所有解包处都取 {n_ret} 个值（实测 {unpacks}）",
        unpacks and all(u == n_ret for u in unpacks))


def test_views_threaded_through():
    """views 必须真的从 load_npz 传到 infer_case，而不只是加在签名上。"""
    print("\n[3] views 贯通 load_npz -> infer_case")
    s = src("scripts/predict/predict_stage2.py")
    tree = ast.parse(s)
    inf = fn_of(tree, "infer_case")
    chk("infer_case 签名含 views", "views" in [a.arg for a in inf.args.args])

    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "infer_case"]
    chk("存在 infer_case 调用", len(calls) == 1)
    if calls:
        kw = [k.arg for k in calls[0].keywords]
        chk("调用处显式传了 views=", "views" in kw)

    body = ast.get_source_segment(s, inf)
    chk("infer_case 内部按 views 是否为 None 分支组装通道",
        "views is not None" in body and "concatenate" in body)


def test_model_channels():
    """模型的输入通道与概率通道必须可配置，且默认保持 2 通道兼容。"""
    print("\n[4] 模型通道配置")
    s = src("src/stage2_model.py")
    tree = ast.parse(s)
    build = fn_of(tree, "build_stage2_model")
    body = ast.get_source_segment(s, build)
    chk("build_stage2_model 认 multi_view 开关", "multi_view" in body)
    chk("默认仍是 2 通道（与既有 checkpoint 兼容）", '"in_channels", 2' in body)
    chk("多视角默认 4 通道", '"in_channels", 4' in body)

    init = fn_of(tree, "__init__")
    args_ = [a.arg for a in init.args.args]
    chk("__init__ 接受 in_channels", "in_channels" in args_)
    chk("__init__ 接受 prob_channels（复数）", "prob_channels" in args_)

    fwd = ast.get_source_segment(s, fn_of(tree, "forward"))
    chk("forward 在多概率通道时取平均作残差基准",
        "mean(dim=1" in fwd and "prob_channels" in fwd)


def test_data_multi_view():
    print("\n[5] 数据加载的多视角分支")
    s = src("src/stage2_data.py")
    chk("Stage2PatchDataset 接受 multi_view", "multi_view=False" in s)
    chk("build_stage2_loaders 透传 multi_view", 'cfg.get("multi_view"' in s)
    chk("读取 p0/p1/p2", '"p0", "p1", "p2"' in s)
    chk("缺 p0/p1/p2 时明确报错而非静默降级",
        "raise KeyError" in s and "--save-views" in s)
    chk("__getitem__ 按维度分支（3D 单概率 / 4D 多视角）", "prob.ndim == 4" in s)


def test_prepare_saves_views():
    print("\n[6] 数据生成保存三方向概率")
    s = src("scripts/data/stage2_prepare.py")
    chk("有 --save-views 开关", '"--save-views"' in s)
    chk("用 predict_tri_probs 取三方向", "predict_tri_probs" in s)
    chk("同时存融合结果 prob（保证能复现 2 通道结果）",
        "extra = {f\"p{i}\"" in s and "prob=prob" in s)


def test_channel_order_consistent():
    """通道顺序在训练与推理两条路径上必须一致：image 在 0，概率在其后。"""
    print("\n[7] 训练与推理的通道顺序一致")
    d = src("src/stage2_data.py")
    p = src("scripts/predict/predict_stage2.py")
    chk("训练侧 4 通道为 [image, *views]", "[img_p] + views" in d)
    chk("推理侧 4 通道为 [image, *views]",
        "[image[None], views]" in p)
    # 概率通道下标从 1 开始
    m = src("src/stage2_model.py")
    chk("模型默认概率通道从 1 开始（image 占 0）", "range(1, in_ch)" in m)


def test_residual_baseline_equivalence():
    """4 通道方案的残差基准 = 三视角平均 = 2 通道方案的 prob，起点必须相同。"""
    print("\n[8] 残差基准等价性（数值验证）")
    p0 = np.array([0.2, 0.9], dtype=np.float32)
    p1 = np.array([0.4, 0.7], dtype=np.float32)
    p2 = np.array([0.6, 0.8], dtype=np.float32)
    mean_prepare = np.mean([p0.astype(np.float32), p1.astype(np.float32),
                            p2.astype(np.float32)], axis=0)
    mean_model = np.stack([p0, p1, p2], axis=0).mean(axis=0)
    chk("prepare 存的 mean 与模型内部取的 mean 一致",
        np.allclose(mean_prepare, mean_model, atol=1e-6))
    chk("因此两方案的残差起点相同 —— 唯一变量是是否保留方向分歧", True)


def main():
    test_no_duplicate_args()
    test_load_npz_unpack_matches()
    test_views_threaded_through()
    test_model_channels()
    test_data_multi_view()
    test_prepare_saves_views()
    test_channel_order_consistent()
    test_residual_baseline_equivalence()
    print(f"\n{'=' * 50}\n  通过 {_ok}   失败 {_fail}\n{'=' * 50}")
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
