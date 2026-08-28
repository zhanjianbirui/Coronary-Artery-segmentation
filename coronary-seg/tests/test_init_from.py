"""--init-from 的行为验证 + BUG-007 机制复现 + 两个训练脚本的接线检查。

不依赖 pytest，直接运行：

    python tests/test_init_from.py

src/ckpt_init.py 只依赖 os 和 torch（不拉 monai），所以能直接 import。
两个训练脚本本身在没装 monai 的机器上 import 不了，所以对它们做 AST 静态检查 ——
重点是确认**两个脚本都真的接上了**，避免只改了一个。
"""
import ast
import io
import os
import sys
import tempfile

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.ckpt_init import validate_init_from, pick_state_dict  # noqa: E402

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


def test_validation(ckpt, src_dir, out_dir):
    V = validate_init_from
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
    check(f"且升到远超初始 lr=3e-4（实测 {lrs[-1]:.2e}，约 {lrs[-1] / 3e-4:.0f} 倍）"
          " —— 会摧毁已训好的权重", lrs[-1] > 3e-4 * 10)


def test_state_dict_compat(ckpt, tmp):
    print("\n[3] pick_state_dict 的兼容性")
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    check("带 'model' 外层时取内层", pick_state_dict(c) is c["model"])

    raw_p = os.path.join(tmp, "raw.pth")
    torch.save({"w": torch.zeros(3)}, raw_p)
    r = torch.load(raw_p, map_location="cpu", weights_only=False)
    check("裸 state_dict（无 'model' 外层）原样返回", pick_state_dict(r) is r)


def test_scripts_wired():
    """两个训练脚本都真的接上了 —— 静态检查，避免只改了一个。"""
    print("\n[4] train.py / train_stage2.py 的接线")
    want = {"validate_init_from", "load_init_weights", "add_init_from_arg"}
    for name in ("train.py", "train_stage2.py"):
        src = io.open(os.path.join(REPO, "scripts", "train", name),
                      encoding="utf-8").read()
        tree = ast.parse(src)
        imported = {
            a.name
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and n.module == "src.ckpt_init"
            for a in n.names
        }
        check(f"{name} 从 src.ckpt_init 导入三个函数", want <= imported)
        check(f"{name} 在 main() 里调用 validate_init_from",
              "validate_init_from(" in src.split("def main(")[-1])
        check(f"{name} 调用 load_init_weights", "load_init_weights(" in src)
        # 关键：init-from 分支必须挂在 resume 的 elif 链上，否则两者可能同时生效
        check(f"{name} 的 init_from 分支挂在 resume 的 elif 链上（保证互斥）",
              'elif cfg.get("init_from")' in src)


def main():
    tmp = tempfile.mkdtemp()
    src_dir = os.path.join(tmp, "exp_tri2p5d")
    out_dir = os.path.join(tmp, "exp_continue")
    os.makedirs(src_dir)
    os.makedirs(out_dir)
    ckpt = os.path.join(src_dir, "best.pth")
    torch.save({"model": {"w": torch.zeros(3)}, "epoch": 59, "val_dice": 0.8135},
               ckpt)

    test_validation(ckpt, src_dir, out_dir)
    test_bug007_mechanism()
    test_state_dict_compat(ckpt, tmp)
    test_scripts_wired()

    print(f"\n{'=' * 52}\n  通过 {_ok}   失败 {_fail}\n{'=' * 52}")
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
