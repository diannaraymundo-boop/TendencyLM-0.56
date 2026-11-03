"""第一步:训练分词器,把原始数据转成训练用的二进制文件。只需要跑一次。

    python prepare_data.py

读取 data/raw/ 下的两份 MiniMind 中文数据,生成:
    data/tokenizer.json      分词器
    data/train.bin           训练数据(两份数据打乱混合)
    data/val.bin             验证数据(留出的 1%,训练时不用)
    data/*.starts.npy        每段对话开头的位置

有工具调用的样本暂时跳过,原型先只处理普通对话。
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from data import TOKEN_MASK, pack
from tokenizer_tool import ASSISTANT_ID, BOS_ID, EOS_ID, SYSTEM_ID, UNK_ID, USER_ID, load_tokenizer, train_tokenizer

ROLE_IDS = {"user": USER_ID, "assistant": ASSISTANT_ID, "system": SYSTEM_ID}


def read_lines(path, limit=0):
    with open(path, encoding="utf-8") as file:
        for i, line in enumerate(file):
            if limit and i >= limit:
                return
            yield line


def parse_record(line):
    """把一行 JSON 转成 [(角色, 文字), ...]。普通文本的角色是 None;跳过的样本返回 None。"""
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    if "text" in record:
        text = record["text"]
        return [(None, text)] if isinstance(text, str) and text.strip() else None
    pieces = []
    for message in record.get("conversations", []):
        role, content = message.get("role"), message.get("content")
        if role not in ROLE_IDS or message.get("tool_calls") or message.get("tools"):
            return None
        if not isinstance(content, str) or not content.strip():
            return None
        pieces.append((role, content))
    return pieces or None


def encode_record(pieces, encoded_texts):
    """拼成一段完整的 token 序列。模型该说的话(回答内容和结束标记)打上来源标记。"""
    parts = [pack([BOS_ID], from_model=False)]
    for (role, _), ids in zip(pieces, encoded_texts):
        if role is None:
            parts.append(pack(ids, from_model=False))
            continue
        parts.append(pack([ROLE_IDS[role]], from_model=False))
        parts.append(pack(ids, from_model=role == "assistant"))
    last_role = pieces[-1][0]
    parts.append(pack([EOS_ID], from_model=last_role == "assistant"))
    return np.concatenate(parts)


class BinWriter:
    """边编码边写文件,不把全部结果留在内存里。"""

    def __init__(self, path):
        self.path = Path(path)
        self.file = open(self.path, "wb")
        self.starts, self.position = [], 0

    def write(self, array):
        self.starts.append(self.position)
        array.tofile(self.file)
        self.position += len(array)

    def close(self):
        self.file.close()
        np.save(self.path.with_suffix(".starts.npy"), np.asarray(self.starts, dtype=np.int64))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--tokenizer_samples", type=int, default=300000, help="训练分词器用多少条样本")
    parser.add_argument("--limit", type=int, default=0, help="每份数据最多读多少行,0 = 全部。试跑时用")
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    started = time.time()
    records, skipped = [], 0
    for name in ("pretrain_t2t_mini.jsonl", "sft_t2t_mini.jsonl"):
        count = 0
        for line in read_lines(Path(args.raw_dir) / name, args.limit):
            record = parse_record(line)
            if record is None:
                skipped += 1
            else:
                records.append(record)
                count += 1
        print(f"{name}:读入 {count} 条", flush=True)
    print(f"共 {len(records)} 条,跳过 {skipped} 条,用时 {time.time() - started:.0f} 秒", flush=True)

    random.Random(args.seed).shuffle(records)

    tokenizer_path = Path(args.out_dir) / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = load_tokenizer(tokenizer_path)
        print("分词器已存在,直接使用。要重新训练请先删除 data/tokenizer.json", flush=True)
    else:
        samples = (text for record in records[: args.tokenizer_samples] for _, text in record)
        tokenizer = train_tokenizer(samples, tokenizer_path, args.vocab_size)
        print(f"分词器训练完成,词表大小 {tokenizer.get_vocab_size()},用时 {time.time() - started:.0f} 秒", flush=True)

    # 分批编码,用满 CPU。前面一小部分留作验证集,其余是训练集
    train_writer = BinWriter(Path(args.out_dir) / "train.bin")
    val_writer = BinWriter(Path(args.out_dir) / "val.bin")
    val_count = max(1, int(len(records) * args.val_ratio))
    record_count, unknown = len(records), 0
    batch_records = 5000
    for start in range(0, record_count, batch_records):
        batch = records[start : start + batch_records]
        texts = [text for record in batch for _, text in record]
        ids = [item.ids for item in tokenizer.encode_batch_fast(texts, add_special_tokens=False)]
        cursor = 0
        for offset, record in enumerate(batch):
            array = encode_record(record, ids[cursor : cursor + len(record)])
            cursor += len(record)
            (val_writer if start + offset < val_count else train_writer).write(array)
            unknown += int(np.count_nonzero((array & TOKEN_MASK) == UNK_ID))
            records[start + offset] = None  # 编码完就释放原文
        if (start // batch_records) % 100 == 0:
            print(f"已编码 {start + len(batch)} / {record_count} 条,用时 {time.time() - started:.0f} 秒", flush=True)
    train_writer.close()
    val_writer.close()

    total_tokens = train_writer.position + val_writer.position
    print(f"训练集 {train_writer.position:,} 个 token,验证集 {val_writer.position:,} 个 token", flush=True)
    print(f"未知 token 占比 {unknown / max(1, total_tokens):.4%},平均每条 {total_tokens / record_count:.0f} 个 token", flush=True)
    print(f"完成,用时 {time.time() - started:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
