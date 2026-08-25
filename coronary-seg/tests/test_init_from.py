"""--init-from 的行为验证 + BUG-007 机制复现。

不依赖 pytest，直接运行：

    python tests/test_init_from.py

刻意**不 import scripts/train.py**（它会拉起 monai，登录节点/本地未必装），
而是用 AST 把 validate_init_from 抽出来单独执行 —— 这同时也验证了
该函数是只依赖 os 的纯函数。
"""
import ast
import io
import os
import sys
import tempfile
import types

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_validate_init_from():
    """从 scripts/train.py 里抽出 validate_init_from，避开重量级 import。"""
    src = io.open(os.path.join(REPO, "scripts", "train.py"),
                  encoding="utf-8").read()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "validate_init_from")
    mod = types.ModuleType("_extracted")
    mod.os = os
    exec(compile(ast.Module([fn], []), "<extracted>", "exec"), mod.__dict__)
    return mod.validate_init_from


_ok = _fail = 0


def check(name, cond):
    global _ok, _fail
    if cond:
        _ok += 1
        print(f"  PASS  {name}")
    else:
        _fail += 1
        print(f"  FAIL  {name}")


def raises(fn, frag):
    try:
        fn()
        return False
    except ValueError as e:
        return frag in str(e)


def test_validation(V, ckpt, src_dir, out_dir):
    print("\n[1] validate_init_from 校验规则")
    check("不传 init_from 直接放行", V(None, out_dir, False) is None)
    check("正常用法放行", V(ckpt, out_dir, False) is None)
    check("与 --resume 同用被拒", raises(lambda: V(ckpt, out_dir, True), "不能同用"))
    check("文件不存在被拒", raises(lambda: V(ckpt + ".nope", out_dir, False), "不存在"))
    # 最重要的护栏：源 ckpt 若在 out-dir 内，训练第一次刷新 best.pth 就会覆盖它
    check("源 ckpt 位于 out-dir 内被拒（防覆盖最优权重）",
          raises(lambda: V(ckpt, src_dir, False), "位于 --out-dir 内"))
    check("路径含 . 时经 realpath 归一后仍被拒",
          raises(lambda: V(ckpt, src_dir + os.sep + ".", False), "位于 --out-dir 内"))
    check("同名前缀目录不误判（exp_tri2p5d vs exp_tri2p5d_v2）",
          V(ckpt, src_dir + "_v2", False) is None)


def test_bug007_mechanism():
    """为什么跑满 --epochs 后不能用 --resume 续训。

    实测结论比"行为不可控"严重得多：LR 会冲到初始值的约 98 倍，
    足以摧毁已训练好的权重。
    """
    print("\n[2] BUG-007 机制复现：--resume 为何救不了跑满的训练")
    opt = torch.optim.AdamW(torch.nn.Linear(2, 2).parameters(), lr=3e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=70)
    for _ in range(70):
        opt.step()
        sch.step()
    lr_end = opt.param_groups[0]["lr"]
    state = sch.state_dict()
    check(f"跑满 T_max=70 后 LR 退火到 ~0（实测 {lr_end:.2e}）", lr_end < 1e-6)
    check("scheduler 的 state_dict 里确实存了 T_max", "T_max" in state)

    opt2 = torch.optim.AdamW(torch.nn.Linear(2, 2).parameters(), lr=3e-4)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=120)
    check("新建时 T_max=120", sch2.T_max == 120)
    sch2.load_state_dict(state)
    check("load_state_dict 后 T_max 被静默覆盖回 70 → 命令行 --epochs 120 无效",
          sch2.T_max == 70)

    lrs = []
    for _ in range(10):
        opt2.step()
        sch2.step()
        lrs.append(opt2.param_groups[0]["lr"])
    check(f"继续 step 后 LR 不降反升（余弦进下一周期）{lrs[0]:.2e} → {lrs[-1]:.2e}",
          lrs[-1] > lrs[0])
    check(f"且升到远超初始 lr=3e-4（实测 {lrs[-1]:.2e}，约 {lrs[-1]/3e-4:.0f} 倍）"
          " —— 会摧毁已训好的权重", lrs[-1] > 3e-4 * 10)


def test_state_dict_compat(ckpt, tmp):
    print("\n[3] 权重读取的兼容性（对应 full_train 的 init-from 分支）")
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    picked = c["model"] if isinstance(c, dict) and "model" in c else c
    check("带 'model' 外层时取内层", picked is c["model"])

    raw_p = os.path.join(tmp, "raw.pth")
    torch.save({"w": torch.zeros(3)}, raw_p)
    r = torch.load(raw_p, map_location="cpu", weights_only=False)
    picked2 = r["model"] if isinstance(r, dict) and "model" in r else r
    check("裸 state_dict（无 'model' 外层）原样返回", picked2 is r)


def main():
    V = load_validate_init_from()
    tmp = tempfile.mkdtemp()
    src_dir = os.path.join(tmp, "exp_tri2p5d")
    out_dir = os.path.join(tmp, "exp_continue")
    os.makedirs(src_dir)
    os.makedirs(out_dir)
    ckpt = os.path.join(src_dir, "best.pth")
    torch.save({"model": {"w": torch.zeros(3)}, "epoch": 59, "val_dice": 0.8135},
               ckpt)

    test_validation(V, ckpt, src_dir, out_dir)
    test_bug007_mechanism()
    test_state_dict_compat(ckpt, tmp)

    print(f"\n{'=' * 52}\n  通过 {_ok}   失败 {_fail}\n{'=' * 52}")
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
