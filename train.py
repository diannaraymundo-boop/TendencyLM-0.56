"""第五版训练:一段话整条说完,检查,填空。

    python train.py --model flow --max_tokens 2000000

中途按 Ctrl+C 可以停,每算 eval_every_tokens 个预测验证并存档一次。
结果在 runs/<名字>/:best.pt、last.pt、metrics.csv。
"""
import argparse
import csv
import math
import random
import time
from dataclasses import fields
from pathlib import Path

import torch

from chain import evaluate_passages, run_batch
from common import count_parameters, get_device, load_checkpoint, model_config_of, save_checkpoint, setup_console
from config import ModelConfig, TrainConfig
from generation import encode_user_turn, generate_passage
from model_baselines import build_model
from passage import PassageSampler
from tokenizer_tool import load_tokenizer

SAMPLE_PROMPTS = ["你好,请介绍一下你自己。", "为什么天空是蓝色的?", "写一首关于秋天的诗。"]
VAL_SEED = 2024
# 同规模 Transformer(7 层、宽 128)的训练记录,进度文件里拿来按同样的更新次数对照
REFERENCE_CURVE = Path(__file__).resolve().parent / "results" / "transformer_reference.csv"
COUNTING_BASELINE = 4.85  # 同一批验证段落上,知道上一个字、只数“这个字后面常跟什么”


def reference_at(step):
    """同规模 Transformer 在同样更新次数时的验证损失(线性插值)。没有记录或超出范围返回 None。"""
    try:
        with open(REFERENCE_CURVE, encoding="utf-8") as file:
            points = [(int(row["step"]), float(row["val_loss"])) for row in csv.DictReader(file)]
    except (OSError, KeyError, ValueError):
        return None
    for (left_step, left_loss), (right_step, right_loss) in zip(points, points[1:]):
        if left_step <= step <= right_step:
            return left_loss + (right_loss - left_loss) * (step - left_step) / (right_step - left_step)
    return None


def write_progress(path, step, seen, minutes, max_minutes, result, sample):
    """给用户看的进度文件:每验证一次写一行,越往下越新。"""
    new_file = not path.exists()
    with open(path, "a", encoding="utf-8") as file:
        if new_file:
            file.write("第五版训练进度。每验证一次(约每 100 万个预测)写一行,越往下越新。说话损失越低越好。\n"
                       f"参照:只看词频 6.94;只数“上一个字后面常跟什么” {COUNTING_BASELINE};"
                       "Transformer 一栏是同规模 Transformer 在同样更新次数时的损失(旧验证办法,只能粗略比)。\n\n")
        reference = reference_at(step)
        remaining = f",还剩约 {max(0.0, max_minutes - minutes):.0f} 分钟" if max_minutes else ""
        file.write(f"{time.strftime('%H:%M')}  已练 {minutes:.0f} 分钟{remaining}  更新 {step:,} 次  {seen:,} 个预测\n"
                   f"    说话损失 {result['speak']:.3f}"
                   f"(Transformer 同样次数时 {f'{reference:.2f}' if reference else '没有记录'};只数数 {COUNTING_BASELINE})"
                   f"  说对的字 {result['word_right']:.1%}  填空后说对 {result['after_right']:.1%}"
                   f"  说错的填空改对 {result['fill_fixed']:.1%}\n"
                   f"    回答样例:{sample}\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model_type", default=None, help="flow")

    def to_bool(text):
        return str(text).lower() in ("1", "true", "yes", "on")

    for config in (ModelConfig(), TrainConfig()):
        for field in fields(config):
            kind = to_bool if isinstance(getattr(config, field.name), bool) else type(getattr(config, field.name))
            if f"--{field.name}" not in parser._option_string_actions:
                parser.add_argument(f"--{field.name}", type=kind, default=None)
    args = vars(parser.parse_args())
    model_cfg = ModelConfig(**{f.name: args[f.name] for f in fields(ModelConfig) if args[f.name] is not None})
    train_cfg = TrainConfig(**{f.name: args[f.name] for f in fields(TrainConfig) if args[f.name] is not None})
    return model_cfg, train_cfg


def make_optimizer(model, train_cfg):
    """时钟表针、权重锐度、意识流幅度、偏置和归一化不做权重衰减:第三版的表针就是被衰减压塌的。"""
    decay, no_decay = [], []
    keep = ("log_period", "log_sharpness", "stream_gain", "bias", "norm")
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if any(part in name for part in keep) else decay).append(parameter)
    return torch.optim.AdamW([{"params": decay, "weight_decay": train_cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
                             lr=train_cfg.lr)


def validation(model, data_dir, device, train_cfg, predictions=None):
    """验证抽样固定,不同模型、不同时候算的是同样的段落。"""
    sampler = PassageSampler(Path(data_dir) / "val.bin", 32, model.cfg.flow_context, train_cfg.max_answer, device, seed=VAL_SEED)
    return evaluate_passages(model, sampler, predictions or train_cfg.eval_beats)


@torch.no_grad()
def print_samples(model, tokenizer, device):
    """打印几个问答,返回第一个的“问 → 答”,给进度文件用。"""
    model.eval()
    first = ""
    for prompt in SAMPLE_PROMPTS:
        output_ids, records = generate_passage(model, encode_user_turn(tokenizer, prompt), device, max_new=60, temperature=0)
        filled = sum(len(record.blanks) for record in records)
        answer = tokenizer.decode(output_ids)
        print(f"  问:{prompt}\n  答:{answer}(检查 {len(records)} 次,重填 {filled} 个字)", flush=True)
        if not first:
            first = f"{prompt} → {' '.join(answer.split())[:60]}"
    model.train()
    return first


def main():
    setup_console()
    model_cfg, train_cfg = parse_args()
    if not train_cfg.max_tokens and not train_cfg.max_minutes:
        raise SystemExit("请指定 --max_tokens 或 --max_minutes,否则训练不会停")
    torch.manual_seed(train_cfg.seed)
    device = get_device()

    data_dir = Path(train_cfg.data_dir)
    tokenizer = load_tokenizer(data_dir / "tokenizer.json")
    run_dir = Path(train_cfg.run_dir) / (train_cfg.run_name or model_cfg.model_type)
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = load_checkpoint(run_dir / "last.pt", device) if train_cfg.resume else None
    if payload:
        model_cfg = model_config_of(payload)
    model_cfg.vocab_size = tokenizer.get_vocab_size()
    if model_cfg.model_type != "flow":
        raise SystemExit("这里只训练 flow 模型")

    model = build_model(model_cfg).to(device)
    optimizer = make_optimizer(model, train_cfg)
    print(f"模型 {model_cfg.model_type},参数 {count_parameters(model):,} 个,存到 {run_dir}", flush=True)

    step, minutes_before, best_val, seen = 0, 0.0, math.inf, 0
    if payload:
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        step, minutes_before, best_val = payload["step"], payload["minutes"], payload["best_val"]
        seen = payload.get("predictions", 0)
        print(f"从第 {step} 步、{seen:,} 个预测接着训练", flush=True)

    sampler = PassageSampler(data_dir / "train.bin", train_cfg.beat_batch, model_cfg.flow_context, train_cfg.max_answer,
                             device, seed=train_cfg.seed + step)
    progress_path = run_dir / "进度.txt"
    metrics_path = run_dir / "metrics.csv"
    new_file = not metrics_path.exists()
    metrics_file = open(metrics_path, "a", newline="", encoding="utf-8")
    metrics = csv.writer(metrics_file)
    if new_file:
        metrics.writerow(["step", "minutes", "predictions", "train_loss", "val_loss", "val_word_right", "val_check_right",
                          "val_fill_fixed", "val_after_right", "speak", "confidence", "fill"])

    def elapsed():
        return minutes_before + (time.time() - started) / 60

    def run_eval(train_loss, stats):
        nonlocal best_val
        result = validation(model, data_dir, device, train_cfg)
        print(f"== 验证:第 {step} 步,{seen:,} 个预测,{elapsed():.1f} 分钟  说话损失 {result['speak']:.4f}  "
              f"说对 {result['word_right']:.1%}  检查判断对 {result['check_right']:.1%}  "
              f"说错的填空改对 {result['fill_fixed']:.1%}  填空后说对 {result['after_right']:.1%}", flush=True)
        metrics.writerow([step, f"{elapsed():.2f}", seen, f"{train_loss:.4f}", f"{result['speak']:.4f}",
                          f"{result['word_right']:.4f}", f"{result['check_right']:.4f}", f"{result['fill_fixed']:.4f}",
                          f"{result['after_right']:.4f}", f"{stats.speak:.4f}", f"{stats.confidence:.4f}", f"{stats.fill:.4f}"])
        metrics_file.flush()
        if result["speak"] < best_val:
            best_val = result["speak"]
            save_checkpoint(run_dir / "best.pt", model, optimizer, model_cfg, train_cfg, step, elapsed(), best_val, seen)
        save_checkpoint(run_dir / "last.pt", model, optimizer, model_cfg, train_cfg, step, elapsed(), best_val, seen)
        sample = print_samples(model, tokenizer, device)
        write_progress(progress_path, step, seen, elapsed(), train_cfg.max_minutes, result, sample)

    model.train()
    started = time.time()
    rng = random.Random(train_cfg.seed + step)
    next_eval = (seen // train_cfg.eval_every_tokens + 1) * train_cfg.eval_every_tokens
    log_started, log_loss, log_steps, log_predictions = time.time(), 0.0, 0, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    while True:
        rate = min(1.0, (step + 1) / train_cfg.warmup_steps)
        if train_cfg.lr_decay_tokens:
            progress = min(1.0, seen / train_cfg.lr_decay_tokens)
            rate *= 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = train_cfg.lr * rate

        loss, stats = run_batch(model, sampler.next_batch(), model_cfg, train_cfg, rng)
        if not torch.isfinite(loss):
            print(f"错误:第 {step} 步损失变成了 {loss.item()},训练停止。试试调小学习率 --lr", flush=True)
            metrics_file.close()
            raise SystemExit(1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        if torch.isfinite(gradient_norm):
            optimizer.step()
        else:
            print(f"警告:第 {step} 步梯度不是有限值,跳过这一步", flush=True)
        step += 1
        seen += stats.predictions

        log_loss += stats.loss
        log_steps += 1
        log_predictions += stats.predictions
        train_loss = stats.loss
        if step % train_cfg.log_every == 0:
            speed = log_predictions / (time.time() - log_started)
            memory = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
            reserved = torch.cuda.memory_reserved() / 1024**3 if device.type == "cuda" else 0.0
            train_loss = log_loss / log_steps
            print(f"第 {step} 步  损失 {train_loss:.4f}(说 {stats.speak:.3f} / 信 {stats.confidence:.3f} / 填 {stats.fill:.3f})"
                  f"  说对 {stats.word_right:.1%}  检查判断对 {stats.check_right:.1%}  说错的填空改对 {stats.fill_fixed:.1%}"
                  f"  {speed:,.0f} 预测/秒  显存峰值 {memory:.2f} GB(实际占着 {reserved:.2f})  {elapsed():.1f} 分钟", flush=True)
            log_started, log_loss, log_steps, log_predictions = time.time(), 0.0, 0, 0
            if device.type == "cuda":
                # 每批长短不一,PyTorch 留着的显存会越攒越多。2026-11-03 第二次改后一批要 6.6 GB,
                # 留着的 3 分钟就涨到 8.2 GB,整张卡 9.4/10 GB,溢出就会像第四版那样塌速。定期还回去
                torch.cuda.empty_cache()

        finished = (train_cfg.max_tokens and seen >= train_cfg.max_tokens) or (
            train_cfg.max_minutes and elapsed() >= train_cfg.max_minutes
        )
        if finished or seen >= next_eval:
            run_eval(train_loss, stats)
            next_eval = (seen // train_cfg.eval_every_tokens + 1) * train_cfg.eval_every_tokens
        if finished:
            break

    metrics_file.close()
    print(f"训练结束,一共 {seen:,} 个预测,最低验证说话损失 {best_val:.4f}", flush=True)
    with open(progress_path, "a", encoding="utf-8") as file:
        file.write(f"{time.strftime('%H:%M')}  训练结束:一共更新 {step:,} 次、{seen:,} 个预测、{elapsed():.0f} 分钟,"
                   f"最低说话损失 {best_val:.3f}\n\n")


if __name__ == "__main__":
    main()
