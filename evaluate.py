"""用同一套固定测试比较第五版的几次训练结果。

    python evaluate.py runs/flow runs/flow_no_review

测试内容:
    验证      整段说完(不看答案)、检查、填一次空。说话损失越低越好
    语序      “张三打了李四”和“李四打了张三”问同一个问题,第一拍的分布差多少
    复述      让模型原样重复一句话,和原文的相似度(0~1)
    回答样例  看检查了几次、重填了几个字

验证损失和 Transformer 一样,是看着前面真正的字预测下一个字。但抽段落的办法不同
(答案最长 256、停止符的规则),和 Transformer 按连续窗口算的验证损失只能粗略比。
"""
import argparse
import difflib
from pathlib import Path

import torch

from common import count_parameters, get_device, load_run, setup_console
from config import TrainConfig
from generation import encode_user_turn, generate_passage, pool_report
from train import validation

ORDER_PAIRS = [
    ("张三打了李四。谁被打了?", "李四打了张三。谁被打了?"),
    ("小明把书送给了小红。书现在在谁手里?", "小红把书送给了小明。书现在在谁手里?"),
    ("猫在追狗。谁在逃跑?", "狗在追猫。谁在逃跑?"),
    ("北京在上海的北边。哪个城市更靠南?", "上海在北京的北边。哪个城市更靠南?"),
]

COPY_TEXTS = [
    "今天天气很好。",
    "今天天气很好,我们一起去公园玩,然后回家吃饭。",
    "今天天气很好,我们一起去公园玩,然后回家吃饭。晚上妈妈做了红烧肉,爸爸说味道很不错,我吃了两碗米饭,弟弟吃了三碗。",
]


@torch.no_grad()
def first_logits(model, tokenizer, text, device):
    tokens = torch.tensor([encode_user_turn(tokenizer, text)], device=device)
    return model(tokens, torch.ones_like(tokens, dtype=torch.bool))[0]


def order_test(model, tokenizer, device):
    results = []
    for first, second in ORDER_PAIRS:
        log_p = torch.log_softmax(first_logits(model, tokenizer, first, device)[0], -1)
        log_q = torch.log_softmax(first_logits(model, tokenizer, second, device)[0], -1)
        kl = 0.5 * ((log_p.exp() * (log_p - log_q)).sum() + (log_q.exp() * (log_q - log_p)).sum())
        results.append((first, second, kl.item()))
    return results


def copy_test(model, tokenizer, device):
    results = []
    for text in COPY_TEXTS:
        token_ids = encode_user_turn(tokenizer, f"请原样重复下面这段话:{text}")
        length = len(tokenizer.encode(text, add_special_tokens=False).ids)
        output_ids, records = generate_passage(model, token_ids, device, max_new=length + 10, temperature=0)
        output = tokenizer.decode(output_ids)
        filled = sum(len(record.blanks) for record in records)
        results.append((text, output, difflib.SequenceMatcher(None, text, output).ratio(), filled))
    return results


def main():
    setup_console()
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="一个或多个训练结果目录")
    parser.add_argument("--which", default="best")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--eval_beats", type=int, default=50000, help="验证时算多少个预测")
    args = parser.parse_args()
    device = get_device()

    summary = []
    for run in args.runs:
        torch.manual_seed(0)
        model, cfg, tokenizer, payload = load_run(run, device, args.which, args.data_dir)
        train_cfg = TrainConfig(**{key: value for key, value in payload["train_cfg"].items() if key in TrainConfig.__dataclass_fields__})
        print(f"\n===== {run}(参数 {count_parameters(model):,} 个,第 {payload['step']} 步,"
              f"{payload.get('predictions', 0):,} 个预测,{payload['minutes']:.0f} 分钟)=====", flush=True)
        result = validation(model, args.data_dir, device, train_cfg, args.eval_beats)
        print(f"验证:说话损失 {result['speak']:.4f}  说对 {result['word_right']:.1%}  检查判断对 {result['check_right']:.1%}  "
              f"说错的填空改对 {result['fill_fixed']:.1%}  填空后说对 {result['after_right']:.1%}", flush=True)

        print("语序测试(差别越大越能分清):", flush=True)
        order = order_test(model, tokenizer, device)
        for first, second, kl in order:
            print(f"  {kl:.3f}  {first} / {second}", flush=True)
        order_mean = sum(item[2] for item in order) / len(order)

        print("复述测试:", flush=True)
        copies = copy_test(model, tokenizer, device)
        for text, output, score, filled in copies:
            print(f"  相似度 {score:.2f}  重填 {filled} 个字\n    输出:{output}", flush=True)
        copy_mean = sum(item[2] for item in copies) / len(copies)

        print("回答样例:", flush=True)
        for prompt in ["你好,请介绍一下你自己。", "为什么天空是蓝色的?", "写一首关于秋天的诗。"]:
            report = pool_report(first_logits(model, tokenizer, prompt, device), tokenizer, 10)
            output_ids, records = generate_passage(model, encode_user_turn(tokenizer, prompt), device, max_new=80)
            filled = sum(len(record.blanks) for record in records)
            print(f"  问:{prompt}\n  第一拍池子 {report.size:.0f} 词,最可能:{' '.join(word for word, _ in report.top)}", flush=True)
            print(f"  答:{tokenizer.decode(output_ids)}(检查 {len(records)} 次,重填 {filled} 个字)", flush=True)
        summary.append((Path(run).name, count_parameters(model), payload["minutes"], result, order_mean, copy_mean))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n===== 汇总 =====", flush=True)
    print("名字 | 参数 | 训练分钟 | 说话损失 | 说对 | 填空后说对 | 语序差别 | 复述相似度", flush=True)
    for name, parameters, minutes, result, order_mean, copy_mean in summary:
        print(f"{name} | {parameters:,} | {minutes:.0f} | {result['speak']:.4f} | {result['word_right']:.1%} | "
              f"{result['after_right']:.1%} | {order_mean:.3f} | {copy_mean:.2f}", flush=True)


if __name__ == "__main__":
    main()
