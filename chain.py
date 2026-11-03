"""第五版的训练办法:一段话整条说完,检查,填空。和生成走完全一样的步骤。

一次训练样本:
    1. 输入时注意力完整跑一次
    2. 逐拍说完整段话。每拍翻译出的字和答案对应位置比,算损失。
       每拍加进上下文的新一列 = 意识流原样 + 说出口的字。训练时说出口的字用答案的字,
       这样模型才知道前面真正说了什么,学得到字和字怎么接;生成时用自己说的字。
       模型同时自己选一个字(和生成同样的选法),留给检查
    3. 检查:输入 + 模型自己选的字,注意力完整跑一次,给每个字置信度。
       和答案一样的字教“可信”,不一样的教“不可信”,两类各占一半分量
    4. 填空:说错的字,加上检查标为不可信的字,挖掉重填,其余的字保留。填的字和答案比

训练和生成的区别:训练时有答案,知道前面真正该说什么、哪些字说错了。
生成时说错一个字会接着错下去,这正是检查和填空要改的。
"""
from typing import NamedTuple

import torch
import torch.nn.functional as F

from config import ModelConfig, TrainConfig
from model_flow import FlowModel
from tokenizer_tool import EOS_ID, PAD_ID, SYSTEM_ID, UNK_ID

SPECIAL_LIMIT = SYSTEM_ID  # 编号不超过它的都是特殊标记


class BatchStats(NamedTuple):
    loss: float
    speak: float          # 说话的损失
    confidence: float     # 检查的损失
    fill: float           # 填空的损失
    word_right: float     # 说出的字和答案一样的比例
    check_right: float    # 检查判断对错判断对了的比例
    fill_fixed: float     # 说错的字,填空后改对的比例
    predictions: int      # 说话算了多少个预测


def choose_token(logits, temperature, top_k, allow_special=True):
    """从分数里选一个字。温度 0 = 选最高的。"""
    logits = logits.detach().clone()
    logits[..., UNK_ID] = float("-inf")
    if not allow_special:
        logits[..., : SPECIAL_LIMIT + 1] = float("-inf")
    if temperature <= 0:
        return logits.argmax(-1)
    logits = logits / temperature
    if top_k > 0:
        threshold = logits.topk(min(top_k, logits.size(-1)), -1).values[..., -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    flat = torch.softmax(logits.reshape(-1, logits.size(-1)), -1)
    return torch.multinomial(flat, 1).view(logits.shape[:-1])


def speak_pass(model: FlowModel, inputs, input_mask, beats, said=None, backprop_beats=0, temperature=0.0, top_k=0):
    """说 beats 拍。said: (批, 拍数) 每拍说出口的字,训练和验证时传答案;不传就用模型自己选的字。

    第 t 拍的预测只看得到第 t 拍之前说出口的字,看不到第 t 个字本身。
    返回 (每拍的分数 (批, 拍数, 词表), 每拍模型自己选的字 (批, 拍数))。选字时不选特殊标记,
    这些字只交给检查;说没说停止符看分数就知道。

    为了每拍少发任务(见 README“提速”):每拍的时间和读的查询说话前一次算好;传了 said 时,
    说出口的字事先就知道,一起过推理网络;自己选的字不进下一拍,说完整段再一起选。结果和每拍现算一样,
    只是自己选字时随机抽到的会换一批,抽法不变。
    """
    state = model.prepare(inputs, input_mask)
    plan = model.plan(state, beats)
    heard = model.hear(said[:, :beats]).unbind(1) if said is not None and model.cfg.use_spoken_word else None
    all_logits, words = [], []
    for beat in range(beats):
        logits, stream, _ = model.speak(state, plan)
        all_logits.append(logits)
        if said is None:
            chosen = choose_token(logits, temperature, top_k, allow_special=False)
            words.append(chosen)
            state = model.grow(state, stream, chosen)
        else:
            state = model.grow(state, stream, said[:, beat], None if heard is None else heard[beat])
        if backprop_beats and (beat + 1) % backprop_beats == 0:
            state = model.cut_backprop(state)
    logits = torch.stack(all_logits, 1)
    if said is None:
        return logits, torch.stack(words, 1)
    # 分几段选,免得一下子多占几百 MB 显存(一批 32 段 × 256 拍 × 8000 个词的分数约 260 MB)
    words = [choose_token(part, temperature, top_k, allow_special=False) for part in logits.split(32, 1)]
    return logits, torch.cat(words, 1)


def pick_blanks(need, limit, generator=None):
    """每行最多挑 limit 个要填的位置。need: (批, 字数) True 表示要填。返回 (批, 空数),-1 表示没有。"""
    if not need.any():
        return need.new_full((need.size(0), 0), -1, dtype=torch.long)
    priority = torch.rand(need.shape, device=need.device, generator=generator).masked_fill(~need, 2.0)
    count = min(limit, int(need.sum(1).max()))
    order = priority.argsort(1)[:, :count]
    chosen = need.gather(1, order)
    return torch.where(chosen, order, torch.full_like(order, -1))


def run_batch(model: FlowModel, batch, model_cfg: ModelConfig, cfg: TrainConfig, rng=None):
    """说完一段、检查、填空。返回 (总损失, 统计)。"""
    inputs, input_mask, answers, answer_mask, _ = batch
    device = inputs.device
    logits, spoken = speak_pass(model, inputs, input_mask, answers.size(1), answers, cfg.backprop_beats,
                                cfg.sample_temperature, cfg.sample_top_k)
    predictions = int(answer_mask.sum())
    speak_loss = F.cross_entropy(logits[answer_mask], answers[answer_mask])

    zero = torch.zeros((), device=device)
    confidence_loss = fill_loss = zero
    word_mask = answer_mask & (answers != EOS_ID)
    right = (spoken == answers) & word_mask
    word_right = right.sum().item() / max(1, int(word_mask.sum()))
    check_right = fill_fixed = 0.0
    if model_cfg.use_review and word_mask.any():
        words = spoken.masked_fill(~word_mask, PAD_ID)
        review = model.review(inputs, input_mask, words, word_mask)
        labels = right[word_mask].float()
        weight = None
        if cfg.balance_check:
            right_share = labels.mean().clamp(0.01, 0.99)
            weight = torch.where(labels > 0, 0.5 / right_share, 0.5 / (1 - right_share))
        confidence_loss = F.binary_cross_entropy(review.confidence[word_mask], labels, weight=weight)
        trusted = review.confidence.detach() >= model_cfg.rework_threshold
        check_right = ((trusted == right) & word_mask).sum().item() / int(word_mask.sum())

        wrong = word_mask & ~right
        need = wrong | (word_mask & ~trusted)
        blanks = pick_blanks(need, model_cfg.rework_max_bad)
        if blanks.numel():
            fill_logits = model.fill(review, blanks)
            valid = blanks >= 0
            targets = answers.gather(1, blanks.clamp_min(0))
            fill_loss = F.cross_entropy(fill_logits[valid], targets[valid])
            was_wrong = wrong.gather(1, blanks.clamp_min(0)) & valid
            if was_wrong.any():
                fixed = fill_logits.argmax(-1) == targets
                fill_fixed = (fixed & was_wrong).sum().item() / int(was_wrong.sum())

    total = speak_loss + cfg.confidence_loss_weight * confidence_loss + cfg.fill_loss_weight * fill_loss
    stats = BatchStats(float(total.detach()), float(speak_loss.detach()), float(confidence_loss.detach()),
                       float(fill_loss.detach()), word_right, check_right, fill_fixed, predictions)
    return total, stats


@torch.no_grad()
def evaluate_passages(model: FlowModel, sampler, predictions_wanted, review=True):
    """验证:和训练一样整段说完、检查、填一次空,只是不更新参数、选字用最高分的。

    说话的损失和训练一样:每拍知道前面真正说了什么,预测下一个字。不同模型抽到的是同样的段落,可以直接比。
    返回 {说话损失, 说对的比例, 检查判断对的比例, 说错的字填空后改对的比例, 填空后说对的比例,
          说对的字平均置信度, 说错的字平均置信度}。后两个差得越多,检查越会分辨对错。
    """
    was_training = model.training
    model.eval()
    loss_sum = trust_right = trust_wrong = 0.0
    counted = words = right_total = check_total = wrong_total = fixed_total = after_total = 0
    while counted < predictions_wanted:
        inputs, input_mask, answers, answer_mask, _ = sampler.next_batch()
        logits, spoken = speak_pass(model, inputs, input_mask, answers.size(1), answers)
        loss_sum += F.cross_entropy(logits[answer_mask], answers[answer_mask], reduction="sum").item()
        counted += int(answer_mask.sum())
        word_mask = answer_mask & (answers != EOS_ID)
        right = (spoken == answers) & word_mask
        words += int(word_mask.sum())
        right_total += int(right.sum())
        if not (review and model.cfg.use_review and word_mask.any()):
            continue
        result = model.review(inputs, input_mask, spoken.masked_fill(~word_mask, PAD_ID), word_mask)
        trusted = result.confidence >= model.cfg.rework_threshold
        check_total += int(((trusted == right) & word_mask).sum())
        trust_right += result.confidence[right].sum().item()
        trust_wrong += result.confidence[word_mask & ~right].sum().item()
        blanks = pick_blanks(word_mask & ~trusted, model.cfg.rework_max_bad)
        after = spoken.clone()
        if blanks.numel():
            filled = choose_token(model.fill(result, blanks), 0, 0, allow_special=False)
            valid = blanks >= 0
            rows = torch.arange(blanks.size(0), device=blanks.device).unsqueeze(1).expand_as(blanks)
            after[rows[valid], blanks[valid]] = filled[valid]
        wrong = word_mask & ~right
        fixed = wrong & (after == answers)
        wrong_total += int(wrong.sum())
        fixed_total += int(fixed.sum())
        after_total += int(((after == answers) & word_mask).sum())
    if was_training:
        model.train()
    return {
        "speak": loss_sum / max(1, counted),
        "word_right": right_total / max(1, words),
        "check_right": check_total / max(1, words),
        "fill_fixed": fixed_total / max(1, wrong_total),
        "after_right": after_total / max(1, words),
        "trust_right": trust_right / max(1, right_total),
        "trust_wrong": trust_wrong / max(1, wrong_total),
    }
