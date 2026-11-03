"""检查训练结果可不可信:有没有信息泄漏,模型强在哪类词上。

    python diagnose.py runs/flow
    python diagnose.py runs/flow 另一个存档目录   # 几个存档放在一起,在同一批段落上比

泄漏检查(用训练好的存档)
    填充内容  改掉填充位置的内容,预测必须不变
    传入内容  改掉输入,预测应该变
    答案后半  改掉答案后半段的字,前半段(包括被改的第一个字本身)的预测必须完全不变:
              每拍只能知道之前说出口的字,不能看到要预测的字和后面的字
    单独算    单独算和一批里算结果必须一样

按词拆分说话损失
    第一次出现  这个词在输入和前面的答案里都还没出现过
    新的搭配    词出现过,但没有紧跟在前一个词后面出现过
    重复的搭配  这两个相邻的词已经连着出现过,基本是在抄前文
    特殊标记    停止符等
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from chain import speak_pass
from common import get_device, load_run, setup_console
from passage import PassageSampler
from tokenizer_tool import ASSISTANT_ID, BOS_ID, SYSTEM_ID, USER_ID

KINDS = {"first": "第一次出现", "seen": "新的搭配", "pair": "重复的搭配", "special": "特殊标记"}
MAX_ANSWER = 256


@torch.no_grad()
def leak_check(model, vocab_size, data_dir, device, batch_size=8):
    sampler = PassageSampler(Path(data_dir) / "val.bin", batch_size, model.cfg.flow_context, MAX_ANSWER, device, seed=2024)
    inputs, input_mask, answers, answer_mask, _ = sampler.next_batch()
    beats = answers.size(1)

    def predict(source, mask, said=answers):
        return speak_pass(model, source, mask, beats, said)[0]

    baseline = predict(inputs, input_mask)
    random_tokens = torch.randint(SYSTEM_ID + 1, vocab_size, inputs.shape, device=device)
    padding_change = (predict(torch.where(input_mask, inputs, random_tokens), input_mask) - baseline).abs().max().item()
    inside_change = (predict(torch.where(input_mask, random_tokens, inputs), input_mask) - baseline).abs().max().item()
    half = beats // 2
    changed = answers.clone()
    changed[:, half:] = torch.randint(SYSTEM_ID + 1, vocab_size, changed[:, half:].shape, device=device)
    future_change = (predict(inputs, input_mask, changed)[:, : half + 1] - baseline[:, : half + 1]).abs().max().item()
    single_gap = 0.0
    for row in range(batch_size):
        length = int(input_mask[row].sum())
        alone = speak_pass(model, inputs[row : row + 1, :length], input_mask[row : row + 1, :length], beats, answers[row : row + 1])[0]
        single_gap = max(single_gap, (alone[0] - baseline[row]).abs().max().item())
    return padding_change, inside_change, future_change, single_gap


@torch.no_grad()
def loss_by_kind(model, data_dir, device, batch_size, count, seed=2024):
    sampler = PassageSampler(Path(data_dir) / "val.bin", batch_size, model.cfg.flow_context, MAX_ANSWER, device, seed=seed)
    totals = {kind: [0.0, 0] for kind in KINDS}
    counted = 0
    while counted < count:
        inputs, input_mask, answers, answer_mask, _ = sampler.next_batch()
        logits, _ = speak_pass(model, inputs, input_mask, answers.size(1), answers)
        losses = F.cross_entropy(logits.transpose(1, 2), answers, reduction="none").tolist()
        for row in range(answers.size(0)):
            history = inputs[row][input_mask[row]].tolist()
            for beat, target in enumerate(answers[row].tolist()):
                if not answer_mask[row, beat]:
                    break
                kind = kind_of(target, history)
                totals[kind][0] += losses[row][beat]
                totals[kind][1] += 1
                history.append(target)
                counted += 1
    return totals


def kind_of(target, history):
    if target <= SYSTEM_ID:
        return "special"
    if target not in history:
        return "first"
    return "pair" if (history[-1], target) in zip(history, history[1:]) else "seen"


def external_flags(inputs):
    """对照模型训练时每个 token 带“是不是外部来的”:用户说的、纯文本、角色标记是外部的,回答内容不是。"""
    flags, inside_reply = [], False
    for token in inputs:
        if token in (USER_ID, SYSTEM_ID, BOS_ID, ASSISTANT_ID):
            flags.append(True)
            inside_reply = token == ASSISTANT_ID
        else:
            flags.append(not inside_reply)
    return flags


@torch.no_grad()
def baseline_loss_by_kind(model, data_dir, device, batch_size, count, seed=2024, chunk=128):
    """对照模型(Transformer)在同样的段落上:输入 + 答案连成一串,看着前面的字预测答案里的每个字。
    分块喂进去,每个字最多看前面 tf_context 个 token。抽段落和分类和第五版完全一样。"""
    sampler = PassageSampler(Path(data_dir) / "val.bin", batch_size, model.cfg.flow_context, MAX_ANSWER, device, seed=seed)
    totals = {kind: [0.0, 0] for kind in KINDS}
    counted = 0
    while counted < count:
        inputs, input_mask, answers, answer_mask, in_reply = sampler.next_batch()
        for row in range(answers.size(0)):
            context = inputs[row][input_mask[row]].tolist()
            answer = answers[row][answer_mask[row]].tolist()
            sequence = context + answer
            flags = external_flags(context) + [not bool(in_reply[row])] * len(answer)
            tokens = torch.tensor([sequence], device=device)
            external = torch.tensor([flags], device=device)
            state, pieces = None, []
            for start in range(0, len(sequence), chunk):
                logits, state = model(tokens[:, start : start + chunk], external[:, start : start + chunk], state)
                pieces.append(logits[0])
            logits = torch.cat(pieces)  # 第 i 个位置预测第 i + 1 个 token
            positions = torch.arange(len(context) - 1, len(sequence) - 1, device=device)
            losses = F.cross_entropy(logits[positions], tokens[0, positions + 1], reduction="none").tolist()
            history = list(context)
            for target, loss in zip(answer, losses):
                kind = kind_of(target, history)
                totals[kind][0] += loss
                totals[kind][1] += 1
                history.append(target)
                counted += 1
    return totals


def main():
    setup_console()
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="一个或多个训练结果目录")
    parser.add_argument("--which", default="best")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--eval_beats", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    device = get_device()

    names, results = [], []
    print("===== 泄漏检查 =====", flush=True)
    print("名字 | 改填充的影响(应为 0) | 改输入的影响(应大于 0) | 改答案后半对前半的影响(应为 0) | 单独算和一批算的差别(应接近 0)", flush=True)
    for run in args.runs:
        model, cfg, _, payload = load_run(run, device, args.which, args.data_dir)
        names.append(f"{Path(run).name}({payload['step']:,} 次更新)")
        if cfg.model_type != "flow":
            print(f"{Path(run).name} | 对照模型,不查", flush=True)
            results.append(baseline_loss_by_kind(model, args.data_dir, device, args.batch_size, args.eval_beats))
        else:
            padding_change, inside_change, future_change, single_gap = leak_check(model, cfg.vocab_size, args.data_dir, device)
            print(f"{Path(run).name} | {padding_change:.6f} | {inside_change:.3f} | {future_change:.6f} | {single_gap:.6f}", flush=True)
            results.append(loss_by_kind(model, args.data_dir, device, args.batch_size, args.eval_beats))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n===== 按词拆分的验证说话损失(越低越好)=====", flush=True)
    counts = {kind: results[0][kind][1] for kind in KINDS}
    total = sum(counts.values())
    print("类别 | 占比 | " + " | ".join(names), flush=True)
    for kind, label in KINDS.items():
        row = [f"{result[kind][0] / max(1, result[kind][1]):.4f}" for result in results]
        print(f"{label} | {counts[kind] / total:.1%} | " + " | ".join(row), flush=True)
    overall = [f"{sum(result[kind][0] for kind in KINDS) / total:.4f}" for result in results]
    print("全部 | 100.0% | " + " | ".join(overall), flush=True)


if __name__ == "__main__":
    main()
