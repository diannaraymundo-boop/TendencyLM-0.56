"""几个脚本共用的小工具。"""
import sys
from dataclasses import asdict, fields
from pathlib import Path

import torch

from config import ModelConfig
from model_baselines import build_model
from tokenizer_tool import load_tokenizer


def setup_console():
    """Windows 控制台默认不是 UTF-8,不设置的话中文输出会乱码或报错。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("警告:没有检测到可用的显卡,改用 CPU,会非常慢", flush=True)
    return torch.device("cpu")


def count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def save_checkpoint(path, model, optimizer, model_cfg, train_cfg, step, minutes, best_val, predictions=0):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_cfg": asdict(model_cfg),
        "train_cfg": asdict(train_cfg),
        "step": step,
        "minutes": minutes,
        "best_val": best_val,
        "predictions": predictions,
    }
    # 先写临时文件再改名,训练中途被打断也不会留下损坏的存档
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=True)


# 2026-11-03 以前的存档没有这几项,要按当时的做法读,否则参数对不上
LEGACY_DEFAULTS = {"split_reading": False, "read_paths": 1, "read_by_thought": False, "use_seed": False, "independent_amount": False}


def model_config_of(payload) -> ModelConfig:
    """存档里的模型配置。旧存档多出来的项不要,缺的项用当时的做法(见 LEGACY_DEFAULTS),再缺的用默认值。"""
    known = {field.name for field in fields(ModelConfig)}
    values = {key: value for key, value in payload["model_cfg"].items() if key in known}
    for key, value in LEGACY_DEFAULTS.items():
        values.setdefault(key, value)
    return ModelConfig(**values)


def load_run(run_dir, device, which="best", data_dir="data"):
    """读取一次训练的结果。返回 (模型, 模型配置, 分词器, 存档内容)。"""
    run_dir = Path(run_dir)
    path = run_dir / f"{which}.pt"
    if not path.exists():
        path = run_dir / "last.pt"
    payload = load_checkpoint(path, device)
    cfg = model_config_of(payload)
    model = build_model(cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, cfg, load_tokenizer(Path(data_dir) / "tokenizer.json"), payload
