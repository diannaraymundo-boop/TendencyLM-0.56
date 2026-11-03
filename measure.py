"""实测显存和速度,不需要真实数据,不是训练。改了模型大小后先跑这个。

    python measure.py
    python measure.py --grid

速度按“说话的预测/秒”算:一次心跳一个预测,检查和填空的时间也算在里面。
显存同时看“真正用的峰值”和“PyTorch 实际占着的”:第四版就是只看了前者,
实际占着的把 10 GB 显卡占满后速度塌了。
"""
import argparse
import time

import torch

from chain import run_batch
from common import count_parameters, get_device, setup_console
from config import ModelConfig, TrainConfig
from model_flow import FlowModel


def measure(cfg, train_cfg, batch_size, length, answer, steps=4):
    device = get_device()
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = FlowModel(cfg).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    inputs = torch.randint(10, cfg.vocab_size, (batch_size, length), device=device)
    answers = torch.randint(10, cfg.vocab_size, (batch_size, answer), device=device)
    batch = (inputs, torch.ones_like(inputs, dtype=torch.bool), answers, torch.ones_like(answers, dtype=torch.bool),
             torch.zeros(batch_size, dtype=torch.bool, device=device))
    timings, predictions = [], 0
    for _ in range(steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.time()
        loss, stats = run_batch(model, batch, cfg, train_cfg)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.time() - started)
        predictions = stats.predictions
    per_step = sum(timings[1:]) / max(1, len(timings) - 1)
    peak = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
    reserved = torch.cuda.memory_reserved() / 1024**3 if device.type == "cuda" else 0.0
    del model, optimizer
    return count_parameters(FlowModel(cfg)), peak, reserved, per_step, predictions / per_step


def main():
    setup_console()
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--answer", type=int, default=256)
    parser.add_argument("--vocab", type=int, default=8000)
    parser.add_argument("--grid", action="store_true")
    args = parser.parse_args()
    if torch.cuda.is_available():
        print(f"显卡:{torch.cuda.get_device_name(0)},显存 {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB", flush=True)
    base = ModelConfig(vocab_size=args.vocab)
    combos = [("默认", base, TrainConfig(), args.batch_size, args.length, args.answer)]
    if args.grid:
        combos = [
            ("默认", base, TrainConfig(), 16, 400, 97),
            ("默认", base, TrainConfig(), 16, 512, 256),
            ("默认,一批 32 段", base, TrainConfig(), 32, 512, 256),
            ("旧做法(4+4 轮、共用一张表),一批 32 段",
             ModelConfig(vocab_size=args.vocab, attn_rounds=4, review_rounds=4, split_reading=False, read_paths=1, use_seed=False), TrainConfig(), 32, 512, 256),
            ("意识流 1024 宽", ModelConfig(vocab_size=args.vocab, flow_stream_dim=1024), TrainConfig(), 16, 512, 256),
        ]
    print("名字 | 批 | 输入 | 答案 | 参数 | 每步秒数 | 预测/秒 | 显存峰值 GB | 实际占着 GB", flush=True)
    for name, cfg, train_cfg, batch, length, answer in combos:
        try:
            parameters, peak, reserved, per_step, speed = measure(cfg, train_cfg, batch, length, answer)
            print(f"{name} | {batch} | {length} | {answer} | {parameters:,} | {per_step:.2f} | {speed:,.0f} | {peak:.2f} | {reserved:.2f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"{name} | {batch} | {length} | {answer} | 显存不够", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
