"""和模型对话,或者看一次生成里的检查和填空过程。

    python generate.py --run checkpoints/flow_amount
    python generate.py --run checkpoints/flow_amount --prompt "你好"
    python generate.py --run checkpoints/flow_amount --prompt "你好" --show

--show 显示:
    输入时注意力最看重的词
    每次检查时每个字的置信度,哪些字被挖掉重填
"""
import argparse

import torch

from common import get_device, load_run, setup_console
from generation import effective_count, encode_user_turn, generate_passage
from tokenizer_tool import decode_token


def show_weights(model, tokenizer, token_ids, device, top=8):
    """第一拍每一路读输入的比例,和读意识流起点的比例。旧存档只有一路,就是改倾向的那张权重表。"""
    tokens = torch.tensor([token_ids], device=device)
    with torch.no_grad():
        _, _, weights = model.speak(model.prepare(tokens, torch.ones_like(tokens, dtype=torch.bool)))
    columns = len(token_ids)
    for path, row in enumerate(weights[0]):
        reading = row[:columns]
        values, indices = reading.topk(min(top, columns))
        shown = " ".join(f"{decode_token(tokenizer, token_ids[i])}#{i}({v:.2f})" for v, i in zip(values.tolist(), indices.tolist()))
        seed = f",读起点 {row[columns:].sum().item():.2f}" if row.size(0) > columns else ""
        print(f"    第一拍第 {path + 1} 路读的比例:输入里实际在看约 {effective_count(reading).item():.1f} 个词,最重的:{shown}{seed}",
              flush=True)


def show_checks(tokenizer, records):
    if not records:
        print("    没有检查(说的是空话,或者关掉了检查)", flush=True)
        return
    for number, record in enumerate(records, 1):
        shown = []
        for position, (word, score) in enumerate(zip(record.words, record.confidence)):
            mark = "*" if position in record.blanks else ""
            shown.append(f"{mark}{decode_token(tokenizer, word)}({score:.2f})")
        tail = f",重填 {len(record.blanks)} 个(带 * 的)" if record.blanks else ",没有要重填的"
        print(f"    第 {number} 次检查{tail}:", flush=True)
        print("      " + " ".join(shown), flush=True)


def main():
    setup_console()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--which", default="best")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--max_new", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    device = get_device()
    model, cfg, tokenizer, payload = load_run(args.run, device, args.which, args.data_dir)
    print(f"已加载 {args.run}:第 {payload['step']} 步,验证说话损失 {payload['best_val']:.4f}", flush=True)

    def answer(text, token_ids):
        ids = token_ids + encode_user_turn(tokenizer, text, first_turn=not token_ids)
        ids = ids[-cfg.flow_context :]
        output_ids, records = generate_passage(model, ids, device, args.max_new, args.temperature, args.top_k)
        print(f"模型:{tokenizer.decode(output_ids)}", flush=True)
        if args.show:
            show_weights(model, tokenizer, ids, device)
            show_checks(tokenizer, records)
        return ids + output_ids

    if args.prompt is not None:
        answer(args.prompt, [])
        return
    print("进入对话。输入 new 开始新对话,exit 退出。", flush=True)
    token_ids = []
    while True:
        text = input("你:").strip()
        if text == "exit":
            break
        if text == "new":
            token_ids = []
            continue
        if text:
            token_ids = answer(text, token_ids)


if __name__ == "__main__":
    main()
