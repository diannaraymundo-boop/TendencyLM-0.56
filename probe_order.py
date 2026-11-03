"""排查:语序信息在第五版的哪一步丢了。只读存档,不训练,不改任何东西。

    python probe_order.py checkpoints/flow_amount

拿几组句子:原句 A、人名对调 B、换成别的人名 C(三句长度一样,只在人名的位置不同)。
只在“三句里完全相同的词”上比较,比如“谁”“被”“打”和最后的 <assistant>。这些词本身一样,
差别只能来自上下文:A 和 B 有哪些词完全一样,只是顺序不同;A 和 C 换了人。

在这几步比较:
    注意力特征   注意力 4 轮看完后的结果
    改动         注意力给的倾向方向(还没乘幅度)
    键           权重表往哪看
    叠完倾向     静态意思 + 幅度 × 改动,推理读的就是它
    过推理网络   每个词各自过推理网络后
    第 1 拍合起来、第 1 拍意识流

对调比例 = A 和 B 的差别 ÷ A 和 C 的差别。接近 0:这一步只认得有哪些词,分不出顺序;
接近 1:对调顺序和换人一样被区分开。哪一步突然掉到接近 0,信息就是在那一步丢的。

另外统计验证段落里每个输入词被改动了多少:幅度(权重表定的,旧做法是权重 / 最大权重),倾向长度 ÷ 静态意思长度。
"""
import argparse
from pathlib import Path

import torch

from common import get_device, load_run, setup_console
from generation import encode_user_turn
from model_baselines import build_model
from passage import PassageSampler

GROUPS = [
    ("张三打了李四。谁被打了?", "李四打了张三。谁被打了?", "王五打了赵六。谁被打了?"),
    ("张三被李四打,谁被打了?", "李四被张三打,谁被打了?", "王五被赵六打,谁被打了?"),
    ("小明把书送给了小红。书现在在谁手里?", "小红把书送给了小明。书现在在谁手里?", "老师把书送给了学生。书现在在谁手里?"),
    ("猫在追狗。谁在逃跑?", "狗在追猫。谁在逃跑?", "鸡在追鸭。谁在逃跑?"),
    ("天津在南京的北边。哪个城市更靠南?", "南京在天津的北边。哪个城市更靠南?", "广州在深圳的北边。哪个城市更靠南?"),
    ("老师批评了学生。谁被批评了?", "学生批评了老师。谁被批评了?", "医生批评了孩子。谁被批评了?"),
    ("孩子帮助了医生。谁得到了帮助?", "医生帮助了孩子。谁得到了帮助?", "老师帮助了学生。谁得到了帮助?"),
]

STAGES = [
    ("seed", "意识流起点"),
    ("feature", "注意力特征"),
    ("change", "改动(未乘幅度)"),
    ("keys", "键(往哪看)"),
    ("vectors", "叠完倾向"),
    ("each", "过推理网络"),
    ("merged", "第 1 拍合起来"),
    ("stream", "第 1 拍意识流"),
]


@torch.no_grad()
def stages_of(model, ids, device):
    """按 prepare / speak 的步骤一步步算,把中间结果都留下。"""
    tokens = torch.tensor([ids], device=device)
    mask = torch.ones_like(tokens, dtype=torch.bool)
    meaning = model.embed(tokens)
    now = model.current_time(torch.zeros(1, dtype=torch.long, device=device))
    feature, change, keys, amount, vectors, seed = model._attend(meaning, mask, model.input_attention, now)
    each = model.token_net(vectors)
    state = model.prepare(tokens, mask)
    assert torch.allclose(state.each, each, atol=1e-5), "和 prepare 算的不一样,脚本跟不上模型代码了"
    seed_column = state.beat_each[:, :1] if state.start else None  # 意识流起点(第 0 拍的念头)
    logits, stream, beat_weights = model.speak(state)
    merged = model.read(state)[0]
    beat_weights = beat_weights.mean(1)  # 几路读的比例取平均
    return {
        "feature": feature[0], "change": change[0], "keys": keys[0], "vectors": vectors[0], "each": each[0],
        "merged": merged, "stream": stream, "meaning": meaning[0], "amount": amount[0],
        "tendency": (amount.unsqueeze(-1) * change[:, : tokens.size(1)])[0], "logits": logits[0],
        "beat_weights": beat_weights[0, : tokens.size(1)], "seed": None if seed_column is None else seed_column[0],
    }


def difference(x, y, positions):
    """在这些位置上,两句的差别有多大(相对 x 本身的长度)。"""
    return ((x[positions] - y[positions]).norm(dim=-1).mean() / x[positions].norm(dim=-1).mean().clamp_min(1e-9)).item()


def kl(logits_p, logits_q):
    log_p, log_q = torch.log_softmax(logits_p, -1), torch.log_softmax(logits_q, -1)
    return (0.5 * ((log_p.exp() * (log_p - log_q)).sum() + (log_q.exp() * (log_q - log_p)).sum())).item()


def show_amounts(result, ids, tokenizer, top=6):
    amount, ratio = result["amount"], result["tendency"].norm(dim=-1) / result["meaning"].norm(dim=-1)
    order = amount.argsort(descending=True)[:top].tolist()
    words = "  ".join(f"{tokenizer.id_to_token(ids[i])}#{i} 幅度{amount[i]:.2f} 倾向/本意{ratio[i]:.2f}" for i in order)
    weights = result["beat_weights"]
    top_weight = weights.argsort(descending=True)[:4].tolist()
    reading = "  ".join(f"{tokenizer.id_to_token(ids[i])}#{i} {weights[i]:.2f}" for i in top_weight)
    effective = torch.exp(-(weights * weights.clamp_min(1e-12).log()).sum()).item()
    return words, reading, effective


@torch.no_grad()
def validation_amounts(model, data_dir, device, batches):
    """验证段落上,每个输入词被改动了多少。"""
    sampler = PassageSampler(Path(data_dir) / "val.bin", 32, model.cfg.flow_context, 256, device, seed=2024)
    amounts, ratios, effective, lengths = [], [], [], []
    for _ in range(batches):
        inputs, mask, _, _, _ = sampler.next_batch()
        meaning = model.embed(inputs)
        now = model.current_time(torch.zeros(inputs.size(0), dtype=torch.long, device=device))
        _, change, _, amount, _, _ = model._attend(meaning, mask, model.input_attention, now)
        change = change[:, : inputs.size(1)]
        weights = amount / amount.sum(-1, keepdim=True)  # 只用来数“实际在看几个词”
        ratio = (amount.unsqueeze(-1) * change).norm(dim=-1) / meaning.norm(dim=-1)
        amounts.append(amount[mask])
        ratios.append(ratio[mask])
        effective.append(torch.exp(-(weights * weights.clamp_min(1e-12).log()).sum(-1)))
        lengths.append(mask.sum(1).float())
    return torch.cat(amounts), torch.cat(ratios), torch.cat(effective), torch.cat(lengths)


def main():
    setup_console()
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--which", default="best")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--batches", type=int, default=4, help="统计改动幅度用多少批验证段落(每批 32 段)")
    parser.add_argument("--untrained", action="store_true", help="用同样配置、刚初始化没训练过的模型当尺子")
    args = parser.parse_args()
    device = get_device()
    model, cfg, tokenizer, payload = load_run(args.run, device, args.which, args.data_dir)
    if args.untrained:
        torch.manual_seed(0)
        model = build_model(cfg).to(device).eval()
        print("没训练过的模型(只借用存档的配置)", flush=True)
    else:
        print(f"存档 {args.run}/{args.which}.pt,第 {payload['step']} 步", flush=True)
    print(f"锐度 {model.input_attention.log_sharpness.exp().item():.2f},意识流进下一拍的幅度 {model.stream_gain.item():.3f}\n", flush=True)

    totals = {key: [] for key, _ in STAGES}
    totals["last"] = []
    for first, swapped, replaced in GROUPS:
        ids = [encode_user_turn(tokenizer, text) for text in (first, swapped, replaced)]
        if len({len(item) for item in ids}) != 1:
            print(f"跳过(三句长度不同):{first}", flush=True)
            continue
        shared = [i for i in range(len(ids[0])) if ids[0][i] == ids[1][i] == ids[2][i]]
        a, b, c = (stages_of(model, item, device) for item in ids)
        print(f"===== {first} / {swapped} / {replaced}", flush=True)
        print(f"  共有的词 {len(shared)} 个:{''.join(tokenizer.id_to_token(ids[0][i]) for i in shared[2:])}", flush=True)
        print(f"  {'步骤':<14}{'A和B(对调)':>12}{'A和C(换人)':>12}{'对调比例':>10}", flush=True)
        for key, name in STAGES:
            if a[key] is None:
                continue
            positions = [0] if key in ("merged", "stream", "seed") else shared
            x, y, z = (item[key] for item in (a, b, c))
            ab, ac = difference(x, y, positions), difference(x, z, positions)
            totals[key].append((ab, ac))
            print(f"  {name:<14}{ab:>12.4f}{ac:>12.4f}{ab / max(ac, 1e-9):>10.2f}", flush=True)
        last = [len(ids[0]) - 1]
        ab, ac = difference(a["vectors"], b["vectors"], last), difference(a["vectors"], c["vectors"], last)
        totals["last"].append((ab, ac))
        print(f"  只看最后一个词(<assistant>)叠完倾向:对调 {ab:.4f},换人 {ac:.4f}", flush=True)
        print(f"  第 1 拍分布差别:对调 {kl(a['logits'], b['logits']):.4f},换人 {kl(a['logits'], c['logits']):.4f}", flush=True)
        words, reading, effective = show_amounts(a, ids[0], tokenizer)
        print(f"  A 改动最大的词:{words}", flush=True)
        print(f"  A 第 1 拍读的词(实际在读约 {effective:.1f} 个):{reading}\n", flush=True)

    print("===== 各组平均 =====", flush=True)
    print(f"  {'步骤':<14}{'A和B(对调)':>12}{'A和C(换人)':>12}{'对调比例':>10}", flush=True)
    for key, name in STAGES:
        if not totals[key]:
            continue
        ab = sum(item[0] for item in totals[key]) / len(totals[key])
        ac = sum(item[1] for item in totals[key]) / len(totals[key])
        print(f"  {name:<14}{ab:>12.4f}{ac:>12.4f}{ab / max(ac, 1e-9):>10.2f}", flush=True)

    amounts, ratios, effective, lengths = validation_amounts(model, args.data_dir, device, args.batches)
    print(f"\n===== 验证段落上的改动({len(lengths)} 段,平均输入 {lengths.mean():.0f} 个词)=====", flush=True)
    print(f"  第 1 拍权重表实际在看约 {effective.mean():.1f} 个词(中位 {effective.median():.1f})", flush=True)
    for threshold in (0.5, 0.1, 0.01):
        share = (amounts >= threshold).float().mean().item()
        print(f"  幅度 ≥ {threshold}:{share:.1%} 的词,平均每段 {share * lengths.mean():.1f} 个", flush=True)
    for threshold in (1.0, 0.5, 0.1):
        share = (ratios >= threshold).float().mean().item()
        print(f"  倾向长度 ≥ 静态意思的 {threshold:.0%}:{share:.1%} 的词,平均每段 {share * lengths.mean():.1f} 个", flush=True)
    print(f"  倾向/静态意思:中位 {ratios.median():.4f},平均 {ratios.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
