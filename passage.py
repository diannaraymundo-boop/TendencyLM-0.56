"""第五版的数据读取:一次取一整段“输入 + 应说的话”,不是一次心跳一个随机词。

为什么要这样:第五版说完要检查、填空,一段话必须走完整条链才有意义,
没法像第三版那样随机抽一个词、拿它前面的内容单独算一次心跳。

输入和答案怎么分:
    对话     最后一个 <assistant> 之前的全部内容是输入,之后的回答是答案。
             如果输入里本来就有多轮,之前的回答都算输入,和生成时一致。
    纯文本   随机切一刀,前半是输入,后半是答案。
停止符:
    数据里每段结尾本来就有 <eos>,答案说完整时它就是答案的最后一个词,不再另补。
    回答后面紧跟 <user>(用户接着说话)时,把 <user> 换成 <eos>。
    答案太长被截断时不补停止符:话还没说完,不能教模型在这里停。
"""
from pathlib import Path

import numpy as np
import torch

from data import TOKEN_MASK
from tokenizer_tool import ASSISTANT_ID, EOS_ID, PAD_ID, USER_ID


class PassageSampler:
    """每次取 batch_size 段:输入 (批, 列)、输入 mask、答案 (批, 列)、答案 mask、是否对话。"""

    def __init__(self, path, batch_size, context, max_outputs, device, seed=None):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.starts = np.load(Path(path).with_suffix(".starts.npy"))
        self.batch_size, self.context, self.max_outputs, self.device = batch_size, context, max_outputs, device
        self.rng = np.random.default_rng(seed)

    def _draw_one(self):
        """返回 (输入 token 列表, 答案 token 列表, 是否对话)。"""
        while True:
            index = int(self.rng.integers(len(self.starts)))
            start = int(self.starts[index])
            stop = int(self.starts[index + 1]) if index + 1 < len(self.starts) else len(self.data)
            ids = (np.asarray(self.data[start:stop]) & TOKEN_MASK).astype(np.int64).tolist()
            if len(ids) < 8:
                continue
            reply_positions = [i for i, token in enumerate(ids) if token == ASSISTANT_ID]
            if reply_positions:
                cut = reply_positions[-1] + 1
                inputs, answer, is_reply = ids[:cut], ids[cut:], True
                if USER_ID in answer:
                    answer = answer[: answer.index(USER_ID)] + [EOS_ID]
            else:
                cut = int(self.rng.integers(1, len(ids) - 1))
                inputs, answer, is_reply = ids[:cut], ids[cut:], False
            if len(answer) < 2:
                continue
            return inputs[-self.context :], answer[: self.max_outputs], is_reply

    def next_batch(self):
        samples = [self._draw_one() for _ in range(self.batch_size)]
        widest_input = max(len(item[0]) for item in samples)
        widest_answer = max(len(item[1]) for item in samples)
        inputs = torch.full((self.batch_size, widest_input), PAD_ID, dtype=torch.long)
        answers = torch.full((self.batch_size, widest_answer), PAD_ID, dtype=torch.long)
        input_mask = torch.zeros_like(inputs, dtype=torch.bool)
        answer_mask = torch.zeros_like(answers, dtype=torch.bool)
        in_reply = torch.zeros(self.batch_size, dtype=torch.bool)
        for row, (context, answer, is_reply) in enumerate(samples):
            inputs[row, : len(context)] = torch.tensor(context)
            input_mask[row, : len(context)] = True
            answers[row, : len(answer)] = torch.tensor(answer)
            answer_mask[row, : len(answer)] = True
            in_reply[row] = is_reply
        move = lambda t: t.to(self.device)
        return move(inputs), move(input_mask), move(answers), move(answer_mask), move(in_reply)
