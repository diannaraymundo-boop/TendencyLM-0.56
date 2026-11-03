"""验证对照模型和数据读取。运行:python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import model_config_of
from config import ModelConfig
from data import TokenStream, pack, unpack
from model_baselines import build_model
from passage import PassageSampler
from prepare_data import BinWriter, encode_record
from tokenizer_tool import ASSISTANT_ID, BOS_ID, EOS_ID, PAD_ID, USER_ID

VOCAB = 200


def tiny_config(**overrides):
    values = dict(vocab_size=VOCAB, embed_dim=16, gru_hidden=32, tf_layers=1, tf_heads=2, tf_dim=16, tf_context=48)
    values.update(overrides)
    return ModelConfig(**values)


def random_tokens(batch=2, length=12, seed=0):
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(10, VOCAB, (batch, length), generator=generator)
    tokens[:, 0] = BOS_ID
    return tokens, torch.ones_like(tokens, dtype=torch.bool)


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(0)


# ===== 对照模型 =====


def test_unknown_model_type_is_rejected():
    with pytest.raises(ValueError):
        build_model(tiny_config(model_type="heartbeat"))


def test_old_checkpoint_config_keeps_only_known_settings():
    cfg = model_config_of({"model_cfg": {"model_type": "flow", "flow_dim": 64, "use_append_context": True, "phase_dim": 32}})
    assert cfg.model_type == "flow" and cfg.flow_dim == 64 and not hasattr(cfg, "use_append_context")


@pytest.mark.parametrize("model_type", ["gru", "transformer"])
def test_baseline_prediction_cannot_see_future_tokens(model_type):
    """改掉后面的 token,前面位置的预测必须完全不变,否则损失是假的。"""
    model = build_model(tiny_config(model_type=model_type)).eval()
    tokens, external = random_tokens(batch=2, length=16)
    changed = tokens.clone()
    changed[:, 10:] = torch.randint(10, VOCAB, (2, 6))
    with torch.no_grad():
        a, _ = model(tokens, external)
        b, _ = model(changed, external)
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-6)
    assert not torch.allclose(a[:, 10:], b[:, 10:], atol=1e-6)


@pytest.mark.parametrize("model_type", ["gru", "transformer"])
def test_baselines_keep_context_across_windows(model_type):
    model = build_model(tiny_config(model_type=model_type)).eval()
    tokens, external = random_tokens(length=12)
    with torch.no_grad():
        whole, _ = model(tokens, external)
        first, state = model(tokens[:, :5], external[:, :5])
        second, _ = model(tokens[:, 5:], external[:, 5:], state)
    assert torch.allclose(whole, torch.cat([first, second], 1), atol=1e-4)


# ===== 数据 =====


def test_pack_keeps_token_and_source():
    tokens, external = unpack(np.concatenate([pack([5, 7999], from_model=True), pack([12], from_model=False)]))
    assert tokens.tolist() == [5, 7999, 12]
    assert external.tolist() == [False, False, True]


def test_stream_windows_are_contiguous(tmp_path):
    array = pack(list(range(10, 1000)), from_model=False)
    path = tmp_path / "x.bin"
    array.tofile(path)
    np.save(tmp_path / "x.starts.npy", np.array([0], dtype=np.int64))
    stream = TokenStream(path, batch_size=1, window=10, device="cpu", seed=0)
    inputs_a, _, targets_a = stream.next_window()
    inputs_b, _, _ = stream.next_window()
    assert inputs_a[0, 1:].tolist() == targets_a[0, :-1].tolist()
    assert inputs_b[0, 0] == targets_a[0, -1]


# ===== 第五版的段落读取 =====

# 用真实的编码方式写数据。每段里的 token 都不重复,所以能认出它来自哪里
DIALOGUE = encode_record([("user", ""), ("assistant", ""), ("user", ""), ("assistant", "")], [[10, 11], [20, 21, 22], [12], [30, 31]])
ENDS_WITH_USER = encode_record([("user", ""), ("assistant", ""), ("user", "")], [[13], [50, 51, 52], [14]])
PLAIN = encode_record([(None, "")], [list(range(40, 60))])
DIALOGUE_IDS = unpack(DIALOGUE)[0].tolist()
PLAIN_IDS = unpack(PLAIN)[0].tolist()


def write_records(tmp_path, *records):
    path = tmp_path / "x.bin"
    writer = BinWriter(path)
    for record in records:
        writer.write(record)
    writer.close()
    return path


def draw_all(path, count=200, batch_size=20, context=48, max_outputs=64, seed=0):
    sampler = PassageSampler(path, batch_size, context, max_outputs, "cpu", seed=seed)
    samples = []
    for _ in range(count // batch_size):
        inputs, input_mask, answers, answer_mask, in_reply = sampler.next_batch()
        for row in range(batch_size):
            samples.append((inputs[row][input_mask[row]].tolist(), answers[row][answer_mask[row]].tolist(), bool(in_reply[row])))
    return samples


def test_dialogue_record_looks_like_the_real_format():
    assert DIALOGUE_IDS == [BOS_ID, USER_ID, 10, 11, ASSISTANT_ID, 20, 21, 22, USER_ID, 12, ASSISTANT_ID, 30, 31, EOS_ID]


def test_dialogue_answer_is_the_last_reply_with_exactly_one_stop(tmp_path):
    for inputs, answer, in_reply in draw_all(write_records(tmp_path, DIALOGUE)):
        assert in_reply
        assert inputs == DIALOGUE_IDS[:11]
        assert answer == [30, 31, EOS_ID]


def test_reply_followed_by_user_ends_with_a_stop(tmp_path):
    for inputs, answer, _ in draw_all(write_records(tmp_path, ENDS_WITH_USER)):
        assert answer == [50, 51, 52, EOS_ID]


def test_plain_text_is_cut_once_and_keeps_its_own_stop(tmp_path):
    samples = draw_all(write_records(tmp_path, PLAIN))
    assert len({len(inputs) for inputs, _, _ in samples}) > 1
    for inputs, answer, in_reply in samples:
        assert not in_reply
        assert inputs + answer == PLAIN_IDS
        assert answer.count(EOS_ID) == 1 and answer[-1] == EOS_ID


def test_truncated_answer_gets_no_stop(tmp_path):
    """话没说完就被截断,不能教模型在这里停。"""
    for inputs, answer, _ in draw_all(write_records(tmp_path, PLAIN), max_outputs=4):
        assert len(answer) <= 4
        full = PLAIN_IDS[len(inputs):]
        assert answer == full[:4]
        assert (EOS_ID in answer) == (len(full) <= 4)


def test_short_rows_are_padded_on_the_right(tmp_path):
    sampler = PassageSampler(write_records(tmp_path, DIALOGUE, PLAIN), 16, 48, 64, "cpu", seed=1)
    inputs, input_mask, answers, answer_mask, _ = sampler.next_batch()
    for tokens, mask in ((inputs, input_mask), (answers, answer_mask)):
        lengths = mask.sum(1)
        for row, length in enumerate(lengths.tolist()):
            assert mask[row, :length].all() and not mask[row, length:].any()
            assert (tokens[row, length:] == PAD_ID).all()
