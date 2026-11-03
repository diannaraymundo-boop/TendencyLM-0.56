"""谁被打了:主动句、被动句各一半,名字对调,看回答里先说出的是不是对的人。只读存档,不训练。

    python who_was_hit.py runs/flow_amount

24 句:3 对名字 × 2 个动词(打、批评)× 主动或被动 × 名字对调。选字用最高分的。
判对:回答里先出现的是被打(被批评)的那个人;两个人都没提算错。乱猜约一半。
“说话时”是检查、填空之前说出的话,“填空后”是检查、填空做完的最终回答。
"""
import argparse

import torch

from common import get_device, load_run, setup_console
from generation import encode_user_turn, generate_passage

PAIRS = [("张三", "李四"), ("王五", "赵六"), ("小明", "小红")]
VERBS = [("打了", "打", "谁被打了?"), ("批评了", "批评", "谁被批评了?")]


def sentences():
    """(问句, 对的名字, 另一个名字)。"""
    for a, b in PAIRS:
        for active, passive, question in VERBS:
            yield f"{a}{active}{b}。{question}", b, a
            yield f"{b}{active}{a}。{question}", a, b
            yield f"{a}被{b}{passive}了。{question}", a, b
            yield f"{b}被{a}{passive}了。{question}", b, a


def answered(text, right, other):
    """回答里先出现的是不是对的那个人。"""
    i, j = text.find(right), text.find(other)
    return i >= 0 and (j < 0 or i < j)


def main():
    setup_console()
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--which", default="best")
    parser.add_argument("--data_dir", default="data")
    args = parser.parse_args()
    device = get_device()
    model, _, tokenizer, payload = load_run(args.run, device, args.which, args.data_dir)
    torch.manual_seed(0)
    print(f"存档 {args.run}/{args.which}.pt,第 {payload['step']} 步", flush=True)
    spoken_right = final_right = total = 0
    for prompt, right, other in sentences():
        ids, records = generate_passage(model, encode_user_turn(tokenizer, prompt), device, max_new=40, temperature=0)
        spoken = tokenizer.decode([w for w in records[0].words if w != 0]) if records else tokenizer.decode(ids)
        final = tokenizer.decode(ids)
        total += 1
        spoken_right += answered(spoken, right, other)
        ok = answered(final, right, other)
        final_right += ok
        print(f"{'对' if ok else '错'}  {prompt}(应答 {right})→ {' '.join(final.split())[:50]}", flush=True)
    print(f"\n{total} 句:说话时答对 {spoken_right},填空后答对 {final_right}(乱猜约一半)", flush=True)


if __name__ == "__main__":
    main()
