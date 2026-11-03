"""第五版的小测试:极小的假任务,只检查整条链路学不学得会,不能推断真实效果。

    python small_task.py --task copy --seed 1 --steps 6000
    python small_task.py --task order --seed 1 --steps 3000 --old 1

训练走完整的流程(说话、检查、填空),和真实训练一样。任务:
    copy    复述:用户说 2~6 个词,模型一个个复述,说完说停止符
    order   语序:“A 被 B 打 谁”答 A,“A 打 B 谁”答 B。考的是没见过的名字搭配,只会猜两个名字之一是 50%
    needle  长输入里找特殊词:50~300 个无关词里混着一个只出现一次的特殊词(16 种),说出它。考的时候夹 300 个
    lookup  回原文查表:输入是 6 对“键 值”,答案是随便挑 4 个键、每个键后面跟它的值。键是随便挑的,
            猜不出来;值只有说出键之后回原文找才知道。考的是值说对的比例,只知道有哪 6 个值是 17%

--old 1 用 2026-11-03 以前的做法(读和改倾向共用一张表,注意力 2 轮 + 2 轮),和现在的默认对比。

每隔一段报告(用新抽的题,选字用最高分的):
    说对      说出的字和答案一样的比例
    检查      判断“这个字对不对”判断对的比例;括号里是对的字、错的字平均拿到的置信度,差得越多越会分辨
    填空改对  说错的字里,被检查标出、填空后改对的比例
    填空后    填一次空之后说对的比例
"""
import argparse
import random
import time

import torch

from chain import evaluate_passages, pick_blanks, run_batch, speak_pass
from common import setup_console
from config import ModelConfig, TrainConfig
from generation import generate_passage
from model_flow import FlowModel
from tokenizer_tool import BOS_ID, EOS_ID, PAD_ID, USER_ID

VOCAB = 60
WORD_LOW, WORD_HIGH = 10, 40


def small_config(old=False, **overrides):
    """现在的默认:读和改倾向分开、分 4 路读、有意识流起点,注意力 4 轮 + 4 轮(真实尺寸是 8 + 8,这里同样翻倍)。"""
    values = dict(
        vocab_size=VOCAB, flow_dim=32, flow_time_dim=32, flow_dials=8, flow_heads=4,
        flow_stream_dim=128, flow_context=64, max_outputs=16, attn_rounds=4, review_rounds=4,
    )
    if old:
        values.update(attn_rounds=2, review_rounds=2, split_reading=False, read_paths=1, use_seed=False)
    values.update(overrides)
    return ModelConfig(**values)


def pack(inputs, answers):
    x = torch.full((len(inputs), max(map(len, inputs))), PAD_ID, dtype=torch.long)
    y = torch.full((len(answers), max(map(len, answers))), PAD_ID, dtype=torch.long)
    for row, (context, reply) in enumerate(zip(inputs, answers)):
        x[row, : len(context)] = torch.tensor(context)
        y[row, : len(reply)] = torch.tensor(reply)
    return x, x != PAD_ID, y, y != PAD_ID, None


class CopyTask:
    def __init__(self, seed, batch=8, test=False):
        self.rng = random.Random(seed)
        self.batch = batch

    def words(self):
        return [self.rng.randrange(WORD_LOW, WORD_HIGH) for _ in range(self.rng.randint(2, 6))]

    def next_batch(self):
        samples = [self.words() for _ in range(self.batch)]
        return pack([[BOS_ID, USER_ID, *words] for words in samples], [[*words, EOS_ID] for words in samples])


class OrderTask:
    """名字 10~29,被 30,打 31,谁 32。名字搭配 (a + b) % 5 == 0 的留着考,训练时见不到。"""

    NAMES = range(10, 30)

    def __init__(self, seed, batch=8, test=False):
        self.rng = random.Random(seed)
        self.batch, self.test = batch, test

    def pair(self):
        while True:
            a, b = self.rng.sample(self.NAMES, 2)
            if ((a + b) % 5 == 0) == self.test:
                return a, b

    def next_batch(self):
        inputs, answers = [], []
        for _ in range(self.batch):
            a, b = self.pair()
            if self.rng.random() < 0.5:
                inputs.append([BOS_ID, USER_ID, a, 30, b, 31, 32])  # A 被 B 打 谁 → A
                answers.append([a, EOS_ID])
            else:
                inputs.append([BOS_ID, USER_ID, a, 31, b, 32])  # A 打 B 谁 → B
                answers.append([b, EOS_ID])
        return pack(inputs, answers)


class NeedleTask:
    """特殊词 10~25,无关词 40~59,最后问一句 32。"""

    def __init__(self, seed, batch=8, test=False):
        self.rng = random.Random(seed)
        self.batch, self.test = batch, test

    def next_batch(self):
        inputs, answers = [], []
        for _ in range(self.batch):
            count = 300 if self.test else self.rng.randint(50, 300)
            words = [self.rng.randrange(40, 60) for _ in range(count)]
            special = self.rng.randrange(10, 26)
            words.insert(self.rng.randrange(count + 1), special)
            inputs.append([BOS_ID, USER_ID, *words, 32])
            answers.append([special, EOS_ID])
        return pack(inputs, answers)


class LookupTask:
    """键 10~29,值 30~49,分隔 50。答案:4 个键,每个后面跟它的值。"""

    PAIRS, ASKED = 6, 4

    def __init__(self, seed, batch=8, test=False):
        self.rng = random.Random(seed)
        self.batch = batch

    def next_batch(self):
        inputs, answers = [], []
        for _ in range(self.batch):
            keys, values = self.rng.sample(range(10, 30), self.PAIRS), self.rng.sample(range(30, 50), self.PAIRS)
            inputs.append([BOS_ID, USER_ID, *[word for pair in zip(keys, values) for word in pair], 50])
            asked = self.rng.sample(range(self.PAIRS), self.ASKED)
            answers.append([word for index in asked for word in (keys[index], values[index])] + [EOS_ID])
        return pack(inputs, answers)


TASKS = {"copy": CopyTask, "order": OrderTask, "needle": NeedleTask, "lookup": LookupTask}


def copy_rate(model, task, rounds=50, max_rework=0):
    right = 0
    for _ in range(rounds):
        words = task.words()
        output, _ = generate_passage(model, [BOS_ID, USER_ID, *words], "cpu", max_new=len(words) + 3,
                                     temperature=0, max_rework=max_rework)
        right += output == words
    return right / rounds


@torch.no_grad()
def lookup_rate(model, task, batches=50):
    """值说对的比例:说话时(每拍听答案的字)和检查、填一次空之后。"""
    model.eval()
    before = after = total = 0
    for _ in range(batches):
        x, x_mask, y, y_mask, _ = task.next_batch()
        _, spoken = speak_pass(model, x, x_mask, y.size(1), y)
        values = torch.zeros_like(y_mask)
        values[:, 1 : 2 * LookupTask.ASKED : 2] = True
        word_mask = y_mask & (y != EOS_ID)
        review = model.review(x, x_mask, spoken.masked_fill(~word_mask, PAD_ID), word_mask)
        blanks = pick_blanks(word_mask & (review.confidence < model.cfg.rework_threshold), model.cfg.rework_max_bad)
        filled = spoken.clone()
        if blanks.numel():
            chosen = model.fill(review, blanks).argmax(-1)
            valid = blanks >= 0
            rows = torch.arange(blanks.size(0)).unsqueeze(1).expand_as(blanks)
            filled[rows[valid], blanks[valid]] = chosen[valid]
        before += int(((spoken == y) & values).sum())
        after += int(((filled == y) & values).sum())
        total += int(values.sum())
    model.train()
    return before / total, after / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="copy", choices=list(TASKS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--report_every", type=int, default=500)
    parser.add_argument("--balance_check", type=int, default=1, help="1 = 教检查时对、错各占一半分量")
    parser.add_argument("--old", type=int, default=0, help="1 = 2026-11-03 以前的做法")
    parser.add_argument("--read_by_thought", type=int, default=0, help="1 = 读的比例也看当前在想什么")
    parser.add_argument("--threads", type=int, default=0, help="CPU 线程数,同时跑好几个时调小")
    parser.add_argument("--rounds", type=int, default=0, help="注意力轮数(输入和检查一样),0 = 按 --old 定")
    parser.add_argument("--read_paths", type=int, default=0, help="分几路读,0 = 按 --old 定")
    parser.add_argument("--split_reading", type=int, default=-1, help="-1 = 按 --old 定")
    parser.add_argument("--self_can_skip", type=int, default=-1, help="-1 = 用默认")
    parser.add_argument("--pairs", type=int, default=6, help="lookup:输入里有几对")
    parser.add_argument("--asked", type=int, default=4, help="lookup:答案问几个键")
    args = parser.parse_args()
    setup_console()
    if args.threads:
        torch.set_num_threads(args.threads)

    overrides = {"flow_context": 320} if args.task == "needle" else {}
    if args.read_by_thought:
        overrides["read_by_thought"] = True
    if args.rounds:
        overrides.update(attn_rounds=args.rounds, review_rounds=args.rounds)
    if args.read_paths:
        overrides["read_paths"] = args.read_paths
    if args.split_reading >= 0:
        overrides["split_reading"] = bool(args.split_reading)
    if args.self_can_skip >= 0:
        overrides["self_can_skip"] = bool(args.self_can_skip)
    LookupTask.PAIRS, LookupTask.ASKED = args.pairs, args.asked
    torch.manual_seed(args.seed)
    model = FlowModel(small_config(old=bool(args.old), **overrides))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    cfg = TrainConfig(balance_check=bool(args.balance_check))
    Task = TASKS[args.task]
    task = Task(args.seed, args.batch)
    c = model.cfg
    print(f"任务 {args.task},种子 {args.seed},注意力 {c.attn_rounds}+{c.review_rounds} 轮,"
          f"{f'读和改倾向分开、分 {c.read_paths} 路读' + (',自己说过的可以不读' if c.self_can_skip else '') if c.split_reading else '读和改倾向共用一张表(旧做法)'}"
          f"{',读的比例看当前所想' if c.read_by_thought else ''},参数 {sum(p.numel() for p in model.parameters()):,}", flush=True)

    started = time.time()
    model.train()
    log_loss, log_steps = 0.0, 0
    for step in range(args.steps):
        loss, stats = run_batch(model, task.next_batch(), model.cfg, cfg)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        log_loss += stats.loss
        log_steps += 1
        if (step + 1) % args.report_every == 0:
            result = evaluate_passages(model, Task(10_000 + step, test=True), 400)
            extra = ""
            if args.task == "lookup":
                before, after = lookup_rate(model, Task(20_000 + step), 10)
                extra = f"  值说对 {before:.0%}  填空后 {after:.0%}"
            print(f"第 {step + 1} 步  损失 {log_loss / log_steps:.3f}  说话损失 {result['speak']:.3f}  "
                  f"说对 {result['word_right']:.0%}  检查 {result['check_right']:.0%}"
                  f"(对的字 {result['trust_right']:.2f} / 错的字 {result['trust_wrong']:.2f})  "
                  f"填空改对 {result['fill_fixed']:.0%}  填空后 {result['after_right']:.0%}{extra}  {time.time() - started:.0f} 秒",
                  flush=True)
            log_loss, log_steps = 0.0, 0

    model.eval()
    if args.task == "copy":
        final = (f"放开复述整句全对:不检查 {copy_rate(model, CopyTask(99)):.0%}  "
                 f"检查填空 {copy_rate(model, CopyTask(99), max_rework=model.cfg.max_rework_rounds):.0%}")
    elif args.task == "lookup":
        before, after = lookup_rate(model, LookupTask(99))
        final = f"值说对 {before:.0%},检查填空后 {after:.0%}"
    else:
        result = evaluate_passages(model, Task(99, test=True), 2000)
        final = f"考题答对 {result['word_right']:.0%},检查填空后 {result['after_right']:.0%}"
    print(f"结果:任务 {args.task}  种子 {args.seed}  {final}  用时 {time.time() - started:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
