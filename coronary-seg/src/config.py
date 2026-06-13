"""配置加载与管理.

设计:
- YAML 提供默认值, 解析成嵌套 dataclass, 享受类型提示和 IDE 补全.
- 支持 dotted-key 覆盖 (如 train.lr=0.001), 方便命令行临时改超参.
- to_dict() 用于存进 checkpoint, 保证实验可复现.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataCfg:
    data_root: str = ""
    split_json: str = "splits/split.json"
    cache_rate: float = 0.0
    num_workers: int = 8


@dataclass
class PreprocessCfg:
    a_min: float = -200.0
    a_max: float = 800.0
    clip: bool = True
    target_spacing: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    patch_size: list[int] = field(default_factory=lambda: [128, 128, 128])


@dataclass
class TrainCfg:
    max_epochs: int = 500
    batch_size: int = 2
    samples_per_image: int = 2
    pos_ratio: float = 0.8
    lr: float = 2e-4
    weight_decay: float = 1e-5
    amp: bool = True
    val_interval: int = 5
    grad_clip: float = 12.0


@dataclass
class ModelCfg:
    name: str = "unet"
    in_channels: int = 1
    out_channels: int = 2
    channels: list[int] = field(default_factory=lambda: [16, 32, 64, 128, 256])
    strides: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    num_res_units: int = 2
    dropout: float = 0.0
    # --- SegResNet 专用 (name=segresnet 时生效) ---
    init_filters: int = 32
    blocks_down: list[int] = field(default_factory=lambda: [1, 2, 2, 4])
    blocks_up: list[int] = field(default_factory=lambda: [1, 1, 1])


@dataclass
class LossCfg:
    name: str = "dice_ce_cldice"      # dice_ce | dice_ce_cldice
    cldice_weight: float = 0.5        # clDice 项权重 (总损失 = DiceCE + w*clDice)
    cldice_iters: int = 3             # 软骨架迭代次数, 越大越准越慢


@dataclass
class InferCfg:
    sw_batch_size: int = 4
    overlap: float = 0.5
    tta: bool = True                  # 测试时增强 (8 向翻转平均), 推理慢 8 倍但更准
    min_component_voxels: int = 30    # 后处理: 删除小于此体素数的连通域(去碎片假阳性), 0=关闭


@dataclass
class OutputCfg:
    work_dir: str = "runs/exp_default"


@dataclass
class Config:
    project_name: str = "coronary-seg"
    seed: int = 42
    data: DataCfg = field(default_factory=DataCfg)
    preprocess: PreprocessCfg = field(default_factory=PreprocessCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    infer: InferCfg = field(default_factory=InferCfg)
    output: OutputCfg = field(default_factory=OutputCfg)

    # ---- 构造 ----
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        raw = copy.deepcopy(raw)
        return cls(
            project_name=raw.get("project_name", "coronary-seg"),
            seed=raw.get("seed", 42),
            data=DataCfg(**raw.get("data", {})),
            preprocess=PreprocessCfg(**raw.get("preprocess", {})),
            train=TrainCfg(**raw.get("train", {})),
            model=ModelCfg(**raw.get("model", {})),
            loss=LossCfg(**raw.get("loss", {})),
            infer=InferCfg(**raw.get("infer", {})),
            output=OutputCfg(**raw.get("output", {})),
        )

    # ---- 命令行覆盖: ["train.lr=0.001", "data.num_workers=4"] ----
    def apply_overrides(self, overrides: list[str]) -> "Config":
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"覆盖项格式应为 key=value, 收到: {item!r}")
            key, value = item.split("=", 1)
            self._set_dotted(key.strip(), value.strip())
        return self

    def _set_dotted(self, dotted_key: str, value: str) -> None:
        parts = dotted_key.split(".")
        obj: Any = self
        for p in parts[:-1]:
            if not hasattr(obj, p):
                raise KeyError(f"未知配置段: {p} (在 {dotted_key})")
            obj = getattr(obj, p)
        leaf = parts[-1]
        if not hasattr(obj, leaf):
            raise KeyError(f"未知配置项: {leaf} (在 {dotted_key})")
        current = getattr(obj, leaf)
        setattr(obj, leaf, _coerce(value, current))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)


def _coerce(value: str, reference: Any) -> Any:
    """把命令行字符串转成 reference 同类型. 支持 list (逗号分隔)."""
    if isinstance(reference, bool):
        return value.lower() in ("1", "true", "yes", "y", "on")
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    if isinstance(reference, list):
        items = [v for v in value.replace("[", "").replace("]", "").split(",") if v != ""]
        if reference and isinstance(reference[0], int):
            return [int(v) for v in items]
        if reference and isinstance(reference[0], float):
            return [float(v) for v in items]
        return items
    return value
