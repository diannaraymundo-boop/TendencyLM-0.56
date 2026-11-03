"""第五版的生成:说一段话,检查,把置信度低的字挖掉重填,置信度高的保留。

和训练用同样的步骤(chain.py 的选字、model 的 speak / review / fill),
唯一的区别:生成时没有答案,哪些字要重填完全靠模型自己的置信度。
"""
import math
from dataclasses import dataclass
from typing import List, Tuple

import torch

from chain import choose_token, pick_blanks
from model_flow import FlowModel
from tokenizer_tool import ASSISTANT_ID, BOS_ID, EOS_ID, PAD_ID, USER_ID, decode_token

STOP_IDS = (EOS_ID, USER_ID, BOS_ID, PAD_ID)


def encode_user_turn(tokenizer, text, first_turn=True) -> List[int]:
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    return ([BOS_ID] if first_turn else []) + [USER_ID] + ids + [ASSISTANT_ID]


@torch.no_grad()
def speak_until_stop(model: FlowModel, tokens, mask, max_new, temperature, top_k):
    """逐拍说,直到全部说出停止符或到上限。返回 (说出的字 (批, 拍数),没说满的补 PAD;每行说了几个字)。"""
    state = model.prepare(tokens, mask)
    plan = model.plan(state, max_new)
    batch = tokens.size(0)
    words = torch.full((batch, max_new), PAD_ID, dtype=torch.long, device=tokens.device)
    live = torch.ones(batch, dtype=torch.bool, device=tokens.device)
    for beat in range(max_new):
        logits, stream, _ = model.speak(state, plan)
        chosen = choose_token(logits, temperature, top_k)
        stop_here = torch.isin(chosen, torch.tensor(STOP_IDS, device=chosen.device)) & live
        live &= ~stop_here
        words[:, beat] = torch.where(live, chosen, torch.full_like(chosen, PAD_ID))
        if not live.any():
            break
        state = model.grow(state, stream, chosen)
    lengths = (words != PAD_ID).sum(1)
    return words[:, : int(lengths.max()) if lengths.numel() else 0], lengths


@dataclass
class CheckRecord:
    words: List[int]          # 这次检查时的字
    confidence: List[float]   # 每个字的置信度
    blanks: List[int]         # 挖掉重填的位置


@torch.no_grad()
def generate_passage(model: FlowModel, token_ids, device, max_new=None, temperature=0.8, top_k=40,
                     threshold=None, max_rework=None) -> Tuple[List[int], List[CheckRecord]]:
    """说一段话,检查,填空,直到没有低置信度的字或到上限。长度在说话时就定了,填空一个空填一个字。"""
    max_new = max_new or model.cfg.max_outputs
    threshold = model.cfg.rework_threshold if threshold is None else threshold
    max_rework = model.cfg.max_rework_rounds if max_rework is None else max_rework
    tokens = torch.tensor([token_ids], device=device)
    mask = torch.ones_like(tokens, dtype=torch.bool)
    words, lengths = speak_until_stop(model, tokens, mask, max_new, temperature, top_k)
    records = []
    if model.cfg.use_review and int(lengths[0]) > 0:
        word_mask = words != PAD_ID
        for _ in range(max_rework + 1):
            review = model.review(tokens, mask, words, word_mask)
            low = (review.confidence < threshold) & word_mask
            blanks = pick_blanks(low, model.cfg.rework_max_bad) if len(records) < max_rework else words[:, :0]
            chosen = blanks[blanks >= 0]
            records.append(CheckRecord(words[0].tolist(), review.confidence[0].tolist(), sorted(chosen.tolist())))
            if chosen.numel() == 0:
                break
            filled = choose_token(model.fill(review, blanks), temperature, top_k, allow_special=False)
            valid = blanks >= 0
            words = words.clone()
            words[0, blanks[valid]] = filled[valid]
    return [item for item in words[0].tolist() if item != PAD_ID], records


# =====================================================================
# 观察工具
# =====================================================================


@dataclass
class PoolReport:
    size: float
    top: List[Tuple[str, float]]


def pool_report(logits, tokenizer, k=20) -> PoolReport:
    log_prob = torch.log_softmax(logits[0], -1)
    prob = log_prob.exp()
    entropy = -(prob * log_prob).sum().item()
    values, indices = prob.topk(k)
    return PoolReport(math.exp(entropy), [(decode_token(tokenizer, i), v) for i, v in zip(indices.tolist(), values.tolist())])


def effective_count(weights) -> torch.Tensor:
    return torch.exp(-(weights * weights.clamp_min(1e-12).log()).sum(-1))
