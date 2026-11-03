"""训练数据的读取。

data/train.bin 和 data/val.bin 是 16 位整数的数组,每个数:低 15 位是 token 编号,
最高位表示来源(0 = 外部,比如用户输入;1 = 模型该说的话)。
"""
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from tokenizer_tool import ASSISTANT_ID, BOS_ID, EOS_ID, PAD_ID, SYSTEM_ID, USER_ID

SOURCE_BIT = 1 << 15
TOKEN_MASK = SOURCE_BIT - 1
ROLE_IDS = (USER_ID, ASSISTANT_ID, SYSTEM_ID)


def pack(token_ids, from_model: bool) -> np.ndarray:
    array = np.asarray(token_ids, dtype=np.uint16)
    return array | SOURCE_BIT if from_model else array


def unpack(array: np.ndarray):
    """返回 (token 编号, 是否来自外部)。"""
    return (array & TOKEN_MASK).astype(np.int64), (array & SOURCE_BIT) == 0


# =====================================================================
# 心跳模型用:每次心跳一个样本
# =====================================================================


class HeartbeatBatch(NamedTuple):
    tokens: torch.Tensor  # (批, 列) 传入 token,短的在右边补填充
    external: torch.Tensor  # (批, 列) 是否来自外部,只有对照模型用
    mask: torch.Tensor  # (批, 列) True 表示真实的 token
    beats: torch.Tensor  # (批,) 这次心跳的时钟偏移
    targets: torch.Tensor  # (批,) 要预测的 token
    in_reply: torch.Tensor  # (批,) True 表示对话里模型该说的话,False 表示纯文本


class HeartbeatSampler:
    """每次心跳一个样本:在数据里随机取一个 token 当预测目标,它前面的内容当传入 token。

    对话:只预测模型该说的话,时钟从这轮回答的第一个词算起。回答结束后紧跟 <user>、没有 <eos> 时,
    把它当成停止符,因为数据里只有整段结尾才有 <eos>。用户说的话不预测。
    纯文本:没有回答,随机切一刀,前半当传入,后半当心跳要预测的内容。传入的内容不变,只有时钟随切的位置变。

    每个样本单独抽,所以固定种子时,不管一批多大,抽到的样本和顺序都一样,不同模型可以直接比。
    """

    def __init__(self, path, batch_size, context, max_beats, device, seed=None):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.batch_size, self.context, self.max_beats, self.device = batch_size, context, max_beats, device
        self.rng = np.random.default_rng(seed)
        self.starts = np.load(Path(path).with_suffix(".starts.npy"))  # 每段对话开头的位置

    def _draw(self):
        """返回 (传入 token 的原始数据, 时钟偏移, 目标, 是否回答)。"""
        while True:
            position = int(self.rng.integers(len(self.data)))
            start = int(self.starts[np.searchsorted(self.starts, position, side="right") - 1])
            if position == start:
                continue  # 对话的开头无法预测
            before = np.asarray(self.data[start:position])
            token, from_model = int(self.data[position]) & TOKEN_MASK, int(self.data[position]) >= SOURCE_BIT

            if int(self.data[start + 1]) & TOKEN_MASK in ROLE_IDS:
                ends_reply = token == USER_ID and before[-1] >= SOURCE_BIT
                if not from_model and not ends_reply:
                    continue
                beats = len(before) - 1 - int(np.flatnonzero(before < SOURCE_BIT)[-1])
                if beats >= self.max_beats:
                    continue
                return before, beats, EOS_ID if ends_reply else token, True

            beats = int(self.rng.integers(min(len(before), self.max_beats)))
            return before, beats, token, False

    def next_batch(self) -> HeartbeatBatch:
        samples = [self._draw() for _ in range(self.batch_size)]
        histories = [unpack(sample[0][-self.context :]) for sample in samples]
        longest = max(len(tokens) for tokens, _ in histories)
        tokens = np.full((self.batch_size, longest), PAD_ID, dtype=np.int64)
        external = np.ones((self.batch_size, longest), dtype=bool)
        mask = np.zeros((self.batch_size, longest), dtype=bool)
        for row, (ids, flags) in enumerate(histories):
            tokens[row, : len(ids)], external[row, : len(ids)], mask[row, : len(ids)] = ids, flags, True
        to_device = lambda array: torch.from_numpy(array).to(self.device)
        return HeartbeatBatch(
            to_device(tokens),
            to_device(external),
            to_device(mask),
            to_device(np.array([sample[1] for sample in samples], dtype=np.int64)),
            to_device(np.array([sample[2] for sample in samples], dtype=np.int64)),
            to_device(np.array([sample[3] for sample in samples], dtype=bool)),
        )


# =====================================================================
# 对照模型训练用:整段窗口
# =====================================================================


class TokenStream:
    """同时开 batch_size 个读取位置,每个位置连续往后读,窗口之间状态可以接着用。

    读到文件末尾就跳到随机一段对话的开头。
    """

    def __init__(self, path, batch_size, window, device, seed=None):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.batch_size, self.window, self.device = batch_size, window, device
        self.rng = np.random.default_rng(seed)
        starts_path = Path(path).with_suffix(".starts.npy")
        self.starts = np.load(starts_path)  # 每段对话开头的位置
        self.positions = self.starts[self.rng.integers(len(self.starts), size=batch_size)].copy()

    def next_window(self):
        """返回 (输入 token, 输入来源, 目标 token),形状都是 (批, 窗口)。"""
        batch = []
        for i in range(self.batch_size):
            if self.positions[i] + self.window + 1 >= len(self.data):
                self.positions[i] = self.starts[self.rng.integers(len(self.starts))]
            start = int(self.positions[i])
            batch.append(self.data[start : start + self.window + 1])
            self.positions[i] += self.window
        tokens, external = unpack(np.stack(batch))
        tokens = torch.from_numpy(tokens).to(self.device)
        external = torch.from_numpy(external).to(self.device)
        return tokens[:, :-1], external[:, :-1], tokens[:, 1:]


def masked_targets(targets):
    """下一段对话的开头无法预测,不计入损失。"""
    return targets.masked_fill(targets == BOS_ID, -100)
