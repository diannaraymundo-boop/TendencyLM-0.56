"""验证第五版真的按用户的链条工作。运行:python -m pytest tests -q

用户的链条:注意力修正输入 → 推理产出意识流 → 翻译成字 → 注意力检查字、覆盖倾向 →
推理填空(置信度高的保留)→ 再翻译、再检查。
"""
import math
import random
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chain import evaluate_passages, pick_blanks, run_batch, speak_pass
from config import ModelConfig, TrainConfig
from generation import generate_passage, speak_until_stop
from model_flow import FlowModel, absolute_positions, positions_to_order
from tokenizer_tool import BOS_ID, EOS_ID, PAD_ID, SYSTEM_ID

VOCAB = 200
DIM = 16
STREAM = 48


def tiny_config(**overrides):
    values = dict(
        vocab_size=VOCAB, flow_dim=DIM, flow_time_dim=16, flow_dials=4, flow_heads=2,
        flow_stream_dim=STREAM, flow_context=48, max_outputs=24, attn_rounds=2, review_rounds=2,
    )
    values.update(overrides)
    return ModelConfig(**values)


def tiny_model(**overrides):
    return FlowModel(tiny_config(**overrides))


def make_batch(batch=2, length=8, answer=5, seed=0):
    """手动造一批:输入 (批, 长度),答案 (批, 答案长度 + 停止符)。"""
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randint(10, VOCAB, (batch, length), generator=generator)
    inputs[:, 0] = BOS_ID
    answers = torch.randint(10, VOCAB, (batch, answer), generator=generator)
    answers = torch.cat([answers, torch.full((batch, 1), EOS_ID)], 1)
    return inputs, torch.ones_like(inputs, dtype=torch.bool), answers, torch.ones_like(answers, dtype=torch.bool), None


def spoken_state(model, inputs, mask, beats):
    """说 beats 拍之后的状态,每拍说出口的字是 60、61、62……"""
    state = model.prepare(inputs, mask)
    for beat in range(beats):
        _, stream, _ = model.speak(state)
        state = model.grow(state, stream, torch.full((inputs.size(0),), 60 + beat))
    return state


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(0)


# ===== 说话:一次心跳一个字,意识流活下去 =====


def test_heartbeat_predicts_one_word_and_the_stream_is_not_compressed():
    model = tiny_model()
    inputs, mask, _, _, _ = make_batch(batch=3)
    logits, stream = model(inputs, mask)
    assert logits.shape == (3, VOCAB)
    assert stream.shape == (3, STREAM) and STREAM > DIM  # 意识流比一个词的意思宽,不压缩


def test_next_heartbeat_gets_both_the_stream_and_the_spoken_word():
    """新的一列 = 意识流原样 + 说出口的字。两样都会影响下一拍。

    刚初始化时所有字的静态意思都很接近,这里先把它们拉开,检查的是有没有这条路。
    """
    model = tiny_model()
    with torch.no_grad():
        model.embed.weight.normal_(0, 1.0)
    inputs, mask, _, _, _ = make_batch(batch=1)
    state = model.prepare(inputs, mask)
    _, stream, _ = model.speak(state)
    word = torch.tensor([60])
    base = model.speak(model.grow(state, stream, word))[0]
    other_stream = model.speak(model.grow(state, torch.randn_like(stream), word))[0]
    other_word = model.speak(model.grow(state, stream, torch.tensor([61])))[0]
    assert not torch.allclose(base, other_stream, atol=1e-5)
    assert not torch.allclose(base, other_word, atol=1e-5)


def test_spoken_word_switch_goes_back_to_stream_only():
    model = tiny_model(use_spoken_word=False)
    inputs, mask, _, _, _ = make_batch(batch=1)
    state = model.prepare(inputs, mask)
    _, stream, _ = model.speak(state)
    a = model.speak(model.grow(state, stream, torch.tensor([60])))[0]
    b = model.speak(model.grow(state, stream, torch.tensor([61])))[0]
    assert torch.equal(a, b)


def test_prediction_cannot_see_the_word_it_is_predicting_or_later_ones():
    """第 t 拍只知道之前说出口的字。改掉第 3 个字及以后,前 3 拍的预测完全不变,之后会变。"""
    model = tiny_model()
    inputs, mask, answers, _, _ = make_batch(batch=2, length=6, answer=6)
    changed = answers.clone()
    changed[:, 2:] = torch.randint(10, VOCAB, changed[:, 2:].shape)
    a = speak_pass(model, inputs, mask, answers.size(1), answers)[0]
    b = speak_pass(model, inputs, mask, answers.size(1), changed)[0]
    assert torch.equal(a[:, :3], b[:, :3])
    assert not torch.allclose(a[:, 3:], b[:, 3:], atol=1e-6)


@pytest.mark.parametrize("overrides", [{}, {"use_seed": True}, {"read_by_thought": True},
                                       {"split_reading": False, "read_paths": 1}])
def test_planning_ahead_gives_the_same_beats_and_gradients_as_working_out_each_beat(overrides):
    """每拍的时间、读的查询说话前一次算好,说出口的字一起过推理网络:说出的分数和倒推的误差
    都和每拍现算一样,只是少发任务。新做法、起点、read_by_thought、旧做法都查。"""
    model = tiny_model(**overrides)
    inputs, mask, answers, _, _ = make_batch(batch=2, length=6, answer=5)

    def beats_and_gradients(planned):
        model.zero_grad()
        if planned:
            logits = speak_pass(model, inputs, mask, answers.size(1), answers)[0]
        else:
            state, each_beat = model.prepare(inputs, mask), []
            for beat in range(answers.size(1)):
                logits, stream, _ = model.speak(state)
                each_beat.append(logits)
                state = model.grow(state, stream, answers[:, beat])
            logits = torch.stack(each_beat, 1)
        F.cross_entropy(logits.flatten(0, 1), answers.flatten()).backward()
        return logits.detach(), {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

    planned_logits, planned_gradients = beats_and_gradients(True)
    each_logits, each_gradients = beats_and_gradients(False)
    assert torch.allclose(planned_logits, each_logits, atol=1e-5)
    assert planned_gradients.keys() == each_gradients.keys()
    for name, gradient in planned_gradients.items():
        assert torch.allclose(gradient, each_gradients[name], atol=1e-6), name


def test_words_for_the_check_are_still_chosen_like_generation():
    """自己选的字说完整段再一起选:温度 0 时就是每拍分数最高的字,不选特殊标记。"""
    model = tiny_model()
    inputs, mask, answers, _, _ = make_batch(batch=2, length=6, answer=40)
    logits, words = speak_pass(model, inputs, mask, answers.size(1), answers)
    expected = logits.clone()
    expected[..., : SYSTEM_ID + 1] = float("-inf")
    assert words.shape == answers.shape
    assert torch.equal(words, expected.argmax(-1))


def test_reading_covers_the_input_and_every_beat_separately():
    """每拍分几路读;每一路在原文上加起来一定是 1,原文不会被自己说过的挤掉。
    说过的列那一份可以选“这拍不读”,加起来不到 1;关掉这个选项时正好是 1。"""
    for can_skip, seed in ((True, True), (False, True), (True, False)):
        model = tiny_model(self_can_skip=can_skip, use_seed=seed)
        inputs, mask, _, _, _ = make_batch(batch=2, length=6)
        state = spoken_state(model, inputs, mask, 3)
        weights = model.speak(state)[2]
        paths, start = model.cfg.read_paths, int(seed)
        assert paths > 1 and weights.shape == (2, paths, 6 + start + 3)
        assert torch.allclose(weights[..., :6].sum(-1), torch.ones(2, paths), atol=1e-5)
        own = weights[..., 6:].sum(-1)
        if can_skip:
            assert (own <= 1 + 1e-6).all() and (own > 0).all() and (own < 1 - 1e-3).any()
        else:
            assert torch.allclose(own, torch.ones(2, paths), atol=1e-5)
        first = model.speak(model.prepare(inputs, mask))[2]  # 第 1 拍:自己这边只有起点(如果有)
        assert first.shape == (2, paths, 6 + start)
        assert torch.allclose(first[..., :6].sum(-1), torch.ones(2, paths), atol=1e-5)


def test_seed_is_the_first_thought_and_carries_the_whole_input():
    """起点被注意力修正后直接成为意识流的第一列:输入任何一个词变了,起点都变;它不进改倾向的权重表,
    倾向不按权重缩小(起点 = 自己 + 整个改动)。关掉开关就没有这一列。"""
    model = tiny_model(use_seed=True)
    inputs, mask, _, _, _ = make_batch(batch=1, length=6)
    state = model.prepare(inputs, mask)
    assert state.start == 1 and state.beat_each.shape == (1, 1, STREAM)
    for position in (1, 3, 5):
        changed = inputs.clone()
        changed[0, position] = 99
        assert not torch.allclose(model.prepare(changed, mask).beat_each, state.beat_each, atol=1e-6)
    now = model.current_time(torch.zeros(1, dtype=torch.long))
    _, change, _, weights, _, seed = model._attend(model.embed(inputs), mask, model.input_attention, now)
    assert weights.shape == (1, 6)
    assert torch.allclose(seed, model.seed + change[:, 6], atol=1e-6)
    assert torch.allclose(state.beat_each[:, 0], model.token_net(seed), atol=1e-6)
    assert tiny_model(use_seed=False).prepare(inputs, mask).beat_each.size(1) == 0


def test_each_word_gets_its_own_amount():
    """幅度各自定:每个词 0~1,不互相抢,可以有很多词同时改得很多;旧做法最受重视的正好是 1、其余按比例。"""
    inputs, mask, _, _, _ = make_batch(batch=2, length=8)
    for independent in (True, False):
        model = tiny_model(independent_amount=independent)
        with torch.no_grad():
            model.input_attention.log_sharpness.fill_(math.log(0.5))  # 夹角差不多时,大家的幅度也差不多
        now = model.current_time(torch.zeros(2, dtype=torch.long))
        _, _, _, amount, _, _ = model._attend(model.embed(inputs), mask, model.input_attention, now)
        assert amount.shape == (2, 8) and (amount >= 0).all() and (amount <= 1 + 1e-6).all()
        if independent:
            assert (amount > 0.3).sum(-1).min() >= 6  # 很多词同时改得不少
            assert (amount.max(-1).values < 1 - 1e-3).all()  # 不会被硬拉到“最重要的改满”
        else:
            assert torch.allclose(amount.max(-1).values, torch.ones(2))


def test_seed_does_not_shift_the_clock_or_positions():
    """起点不算说过的拍:第 1 拍的时钟是 0,说了几拍时钟就走几格,和没有起点时一样。"""
    model = tiny_model(use_seed=True)
    inputs, mask, _, _, _ = make_batch(batch=1, length=6)
    seen = []
    original = model.current_time
    model.current_time = lambda beats: seen.append(beats.flatten()[0].item()) or original(beats)
    spoken_state(model, inputs, mask, 3)
    model.speak(spoken_state(model, inputs, mask, 2))
    model.current_time = original
    assert 1 in seen and 2 in seen and max(seen) == 2


def test_each_path_reads_only_its_own_part():
    """第 p 路只读每列的第 p 段。省显存的算法要和“先切段再读”结果一样。"""
    model = tiny_model()
    reader, paths = model.read_input, model.cfg.read_paths
    query, keys = torch.randn(3, paths, 2, DIM // paths), torch.randn(3, paths, 7, DIM // paths)
    values, mask = torch.randn(3, 7, STREAM), torch.ones(3, 7, dtype=torch.bool)
    mask[1, 5:] = False
    weights = reader.weigh(query, keys, mask)
    parts = values.view(3, 7, paths, STREAM // paths).transpose(1, 2)
    expected = (weights @ parts).permute(0, 2, 1, 3).reshape(3, 2, STREAM)
    assert torch.allclose(reader.read(query, keys, values, mask)[0], expected, atol=1e-5)


def test_skipping_own_columns_reads_nothing_from_them():
    """“不读”的分数很高时,说过的列怎么变,这一拍说的都不变。"""
    model = tiny_model()
    with torch.no_grad():
        model.read_self.skip.bias.fill_(50.0)
    inputs, mask, _, _, _ = make_batch(batch=1, length=6)
    state = spoken_state(model, inputs, mask, 2)
    other = state._replace(beat_each=torch.randn_like(state.beat_each))
    assert torch.allclose(model.speak(state)[0], model.speak(other)[0], atol=1e-5)


def test_old_way_shares_one_table_with_the_tendency():
    """旧做法(旧存档):只有一路,原文和说过的列挤在一张表里分;第 1 拍这张表就是改倾向的权重表。"""
    model = tiny_model(split_reading=False, read_paths=1, use_seed=False, independent_amount=False)
    inputs, mask, _, _, _ = make_batch(batch=2, length=6)
    weights = model.speak(spoken_state(model, inputs, mask, 3))[2]
    assert weights.shape == (2, 1, 6 + 3) and torch.allclose(weights.sum(-1), torch.ones(2, 1), atol=1e-5)
    now = model.current_time(torch.zeros(2, dtype=torch.long))
    _, _, keys = model.input_attention(model.embed(inputs), mask, now)
    table = model.input_attention.weigh((keys @ model.input_attention.query(now, mask.sum(1) - 1).unsqueeze(-1)).squeeze(-1), mask)
    assert torch.allclose(model.speak(model.prepare(inputs, mask))[2][:, 0], table, atol=1e-6)


def test_old_saves_load_the_old_way():
    from common import model_config_of

    saved = {key: value for key, value in vars(tiny_config()).items() if key not in ("split_reading", "read_paths", "read_by_thought", "use_seed", "independent_amount")}
    cfg = model_config_of({"model_cfg": saved})
    assert cfg.split_reading is False and cfg.read_paths == 1 and cfg.read_by_thought is False and cfg.use_seed is False and cfg.independent_amount is False


def test_reading_is_separate_from_the_tendency():
    """改读的比例,输入词的倾向完全不变,说的话会变:读和改倾向是两回事。"""
    model = tiny_model()
    inputs, mask, _, _, _ = make_batch(batch=1, length=6)
    before_state = model.prepare(inputs, mask)
    before = model.speak(before_state)[0]
    with torch.no_grad():
        model.read_input.time_to_query.weight.add_(0.5)
        model.input_attention.read_key.weight.mul_(-1.0)
    after_state = model.prepare(inputs, mask)
    assert torch.equal(before_state.each, after_state.each)
    assert not torch.allclose(before, model.speak(after_state)[0], atol=1e-6)


def test_reading_does_not_look_at_the_current_thought_unless_switched_on():
    """默认读的比例只由时钟、位置和键决定:换掉最新一列的内容(键不变),比例不变。打开开关后会变。"""
    for switch, should_change in ((False, False), (True, True)):
        model = tiny_model(read_by_thought=switch)
        inputs, mask, _, _, _ = make_batch(batch=1, length=6)
        state = spoken_state(model, inputs, mask, 2)
        weights = model.speak(state)[2]
        other = state._replace(beat_each=torch.cat([state.beat_each[:, :-1], torch.randn_like(state.beat_each[:, -1:])], 1))
        changed = not torch.allclose(weights, model.speak(other)[2], atol=1e-6)
        assert changed == should_change, switch


def test_key_rotation_equals_rotating_by_distance_from_the_latest_word():
    """键按自己的位置、查询按现在的位置各转一次,和“每个键按离最新一个词多远来转”完全一样。"""
    model = tiny_model()
    attention = model.input_attention
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[1, 5:] = False
    raw = torch.randn(2, 7, DIM)
    query = torch.randn(2, DIM)
    trick = (attention.place(raw, absolute_positions(mask)) @ attention.rotation(
        F.normalize(query, dim=-1).unsqueeze(1), -(mask.sum(1) - 1).unsqueeze(1)).squeeze(1).unsqueeze(-1)).squeeze(-1)
    direct = (attention.rotation(F.normalize(raw, dim=-1), positions_to_order(mask)) @ F.normalize(query, dim=-1).unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(trick[mask], direct[mask], atol=1e-5)

    # 读的时候分几路,每路也是同样的转法
    reader = model.read_input
    position = torch.tensor([[9]])
    trick = (reader.query(torch.zeros(1, 1, model.cfg.flow_time_dim), position) @ reader.place(raw[:1], absolute_positions(mask[:1])).transpose(-1, -2))
    query = F.normalize(reader.time_to_query(torch.zeros(1, 1, model.cfg.flow_time_dim)).view(1, 1, reader.paths, -1), dim=-1).transpose(1, 2)
    keys = F.normalize(raw[:1].view(1, 7, reader.paths, -1), dim=-1).transpose(1, 2)
    direct = query @ reader.rotation(keys, 9 - absolute_positions(mask[:1]).float()).transpose(-1, -2)
    assert torch.allclose(trick, direct, atol=1e-5)


def test_padding_does_not_matter():
    model = tiny_model()
    inputs, mask, _, _, _ = make_batch(batch=1, length=6)
    padded = torch.cat([inputs, torch.full((1, 4), PAD_ID)], 1)
    padded_mask = torch.cat([mask, torch.zeros(1, 4, dtype=torch.bool)], 1)
    said = torch.tensor([[60, 61, 62, 63]])
    a = speak_pass(model, inputs, mask, 4, said)[0]
    b = speak_pass(model, padded, padded_mask, 4, said)[0]
    assert torch.allclose(a, b, atol=1e-4)


def test_same_words_in_different_order_give_different_prediction():
    model = tiny_model()
    first = torch.tensor([[BOS_ID, 50, 51, 52, 53]])
    second = torch.tensor([[BOS_ID, 52, 50, 53, 51]])
    mask = torch.ones_like(first, dtype=torch.bool)
    assert (model(first, mask)[0] - model(second, mask)[0]).abs().max() > 1e-4


# ===== 时钟:只作为心跳,只通过注意力起作用 =====


def test_clock_switch_makes_time_constant():
    model = tiny_model(use_clock=False)
    assert torch.equal(model.current_time(torch.tensor([0])), model.current_time(torch.tensor([7])))


def test_clock_only_reaches_the_prediction_through_the_reading_weights():
    """说话时固定读的比例,时钟再怎么走,预测都不变;不固定时会变(证明这个测试抓得住)。"""
    model = tiny_model()
    inputs, mask, _, _, _ = make_batch(batch=1)
    state = spoken_state(model, inputs, mask, 2)
    original_time = model.current_time
    moved_time = lambda beats: original_time(beats + 9)

    readers = (model.read_input, model.read_self)
    fixed = {}
    for reader in readers:
        reader.weigh = lambda q, k, m, s=None, reader=reader: fixed.setdefault(reader, type(reader).weigh(reader, q, k, m, s))
    a = model.speak(state)[0]
    model.current_time = moved_time
    b = model.speak(state)[0]
    for reader in readers:
        del reader.weigh
    c = model.speak(state)[0]
    model.current_time = original_time
    assert len(fixed) == 2
    assert torch.equal(a, b)
    assert not torch.allclose(a, c, atol=1e-6)


# ===== 检查:看字,覆盖倾向,给置信度 =====


def test_check_scores_only_output_words_and_ignores_padding():
    model = tiny_model()
    inputs, mask, _, _, _ = make_batch(batch=1, length=6)
    words = torch.tensor([[60, 61, 62]])
    padded = torch.tensor([[60, 61, 62, PAD_ID, PAD_ID]])
    padded_mask = padded != PAD_ID
    a = model.review(inputs, mask, words, words != PAD_ID)
    b = model.review(inputs, mask, padded, padded_mask)
    assert a.confidence.shape == (1, 3)
    assert torch.allclose(a.confidence, b.confidence[:, :3], atol=1e-5)


def test_check_overwrites_the_tendency_from_static_meaning():
    """覆盖不累加:检查后的每个词 = 静态意思 + 这次的倾向,只由当前的字决定。"""
    model = tiny_model()
    inputs, mask, _, _, _ = make_batch(batch=1, length=6)
    words = torch.tensor([[60, 61, 62]])
    first = model.review(inputs, mask, words, words != PAD_ID)
    again = model.review(inputs, mask, words, words != PAD_ID)
    assert torch.equal(first.vectors, again.vectors)
    assert first.vectors.size(1) == 6 + 3 + int(model.cfg.use_seed)  # 有起点时最后一列是检查后的起点
    tendency = first.vectors[:, : 6 + 3] - model.embed(torch.cat([inputs, words], 1))
    assert tendency.abs().sum() > 0
    # 输入词的倾向也被覆盖,和输入时那次不一样
    _, change, keys = model.input_attention(model.embed(inputs), mask, model.current_time(torch.zeros(1, dtype=torch.long)))
    assert not torch.allclose(tendency[:, :6], change, atol=1e-4)


def test_check_reads_words_not_the_stream():
    """检查只看翻译出的字:同样的字,不管说话时意识流是什么,检查结果一样。"""
    model = tiny_model()
    inputs, mask, _, _, _ = make_batch(batch=1, length=6)
    words = torch.tensor([[60, 61, 62]])
    before = model.review(inputs, mask, words, words != PAD_ID).confidence
    with torch.no_grad():
        model.merge_net[0].weight.add_(1.0)  # 意识流全变了
    after = model.review(inputs, mask, words, words != PAD_ID).confidence
    assert torch.equal(before, after)


# ===== 填空:置信度高的保留,前后的字都看得见 =====


def test_fill_sees_words_before_and_after_the_blank():
    model = tiny_model()
    inputs, mask, _, _, _ = make_batch(batch=1, length=6)
    blanks = torch.tensor([[1]])
    words = torch.tensor([[60, 61, 62, 63]])
    other_after = torch.tensor([[60, 61, 62, 99]])
    other_before = torch.tensor([[77, 61, 62, 63]])
    base = model.fill(model.review(inputs, mask, words, words != PAD_ID), blanks)
    after = model.fill(model.review(inputs, mask, other_after, other_after != PAD_ID), blanks)
    before = model.fill(model.review(inputs, mask, other_before, other_before != PAD_ID), blanks)
    assert not torch.allclose(base, after, atol=1e-6)
    assert not torch.allclose(base, before, atol=1e-6)


def test_fill_uses_the_same_reasoning_and_language_as_speaking():
    model = tiny_model()
    inputs, mask, _, _, _ = make_batch(batch=1, length=6)
    words = torch.tensor([[60, 61, 62]])
    review = model.review(inputs, mask, words, words != PAD_ID)
    speak_before, fill_before = model(inputs, mask)[0], model.fill(review, torch.tensor([[0]]))
    with torch.no_grad():
        model.translate_net.weight.mul_(2.0)
    assert not torch.allclose(speak_before, model(inputs, mask)[0])
    assert not torch.allclose(fill_before, model.fill(review, torch.tensor([[0]])))


def test_pick_blanks_takes_only_marked_positions_up_to_the_limit():
    need = torch.tensor([[True, False, True, True], [False, False, False, False]])
    blanks = pick_blanks(need, 2)
    assert blanks.shape == (2, 2)
    assert set(blanks[0].tolist()) <= {0, 2, 3} and -1 not in blanks[0].tolist()
    assert blanks[1].tolist() == [-1, -1]


# ===== 生成 =====


def test_generation_stops_and_adds_nothing_after_the_stop_symbol():
    model = tiny_model()
    with torch.no_grad():
        model.out_bias.fill_(-100.0)
        model.out_bias[EOS_ID] = 100.0
    tokens, records = generate_passage(model, [BOS_ID, 50, 51], "cpu", max_new=8, temperature=0)
    assert tokens == [] and records == []


def test_generation_keeps_trusted_words_and_refills_the_rest_one_for_one():
    model = tiny_model()
    with torch.no_grad():
        model.out_bias.fill_(-100.0)
        model.out_bias[77] = 100.0  # 说话一直说 77,说满 6 个
        model.confidence_head.weight.zero_()
        model.confidence_head.bias.fill_(-20.0)  # 全部不可信
    tokens, records = generate_passage(model, [BOS_ID, 50, 51], "cpu", max_new=6, temperature=0, max_rework=2)
    assert len(tokens) == 6  # 一个空填一个字,长度不变
    assert len(records) == 3 and records[0].blanks == list(range(6)) and records[-1].blanks == []

    with torch.no_grad():
        model.confidence_head.bias.fill_(20.0)  # 全部可信
    tokens, records = generate_passage(model, [BOS_ID, 50, 51], "cpu", max_new=6, temperature=0, max_rework=2)
    assert tokens == [77] * 6 and len(records) == 1 and records[0].blanks == []


def test_speak_until_stop_handles_rows_that_stop_at_different_times():
    model = tiny_model()
    inputs, mask, _, _, _ = make_batch(batch=3, length=5)
    words, lengths = speak_until_stop(model, inputs, mask, 6, 1.0, 0)
    for row, length in enumerate(lengths.tolist()):
        assert (words[row, :length] != PAD_ID).all() and (words[row, length:] == PAD_ID).all()


# ===== 训练和验证 =====


@pytest.mark.parametrize("overrides", [{}, {"read_by_thought": True}, {"use_seed": True}, {"independent_amount": False}, {"split_reading": False, "read_paths": 1, "use_seed": False, "independent_amount": False}])
def test_training_step_is_finite_and_reaches_every_part(overrides):
    model = tiny_model(**overrides)
    batch = make_batch(batch=3, length=7, answer=4)
    loss, stats = run_batch(model, batch, model.cfg, TrainConfig(backprop_beats=2))
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0, name
    assert stats.predictions == 3 * 5


def test_training_respects_the_answer_length():
    """一批里答案长短不一,短的说完后的位置不算损失,也不交给检查。"""
    model = tiny_model()
    inputs, mask, answers, answer_mask, _ = make_batch(batch=2, length=6, answer=5)
    answer_mask[1, 3:] = False
    answers[1, 3:] = PAD_ID
    _, stats = run_batch(model, (inputs, mask, answers, answer_mask, None), model.cfg, TrainConfig())
    assert stats.predictions == 6 + 3


def test_nothing_leaks_from_the_answer_on_random_data():
    """完全随机、猜不出来的数据上训练一会儿:验证损失必须仍接近乱猜。

    以前的验证把要预测的字本身当念头读出,这里会降到 0.5 左右。答案里不放停止符,它是猜得到的。
    """
    def random_batch(seed):
        inputs, mask, answers, answer_mask, _ = make_batch(batch=8, length=6, answer=6, seed=seed)
        return inputs, mask, answers[:, :-1], answer_mask[:, :-1], None

    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for step in range(120):
        loss, _ = run_batch(model, random_batch(step), model.cfg, TrainConfig())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    class RandomSampler:
        seed = 1000

        def next_batch(self):
            self.seed += 1
            return random_batch(self.seed)

    result = evaluate_passages(model, RandomSampler(), 400)
    assert result["speak"] > math.log(VOCAB) - 0.5


def test_training_lowers_loss_on_a_copy_task():
    """极小的复述任务:输入 A B C,答案 A B C 停止符。损失要明显下降。"""
    model = tiny_model(max_outputs=12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    inputs = torch.tensor([[BOS_ID, 40, 41, 42]])
    answers = torch.tensor([[40, 41, 42, EOS_ID]])
    batch = (inputs, torch.ones_like(inputs, dtype=torch.bool), answers, torch.ones_like(answers, dtype=torch.bool), None)
    first = float(run_batch(model, batch, model.cfg, TrainConfig())[1].speak)
    for _ in range(60):
        loss, _ = run_batch(model, batch, model.cfg, TrainConfig())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert run_batch(model, batch, model.cfg, TrainConfig())[1].speak < first * 0.5
