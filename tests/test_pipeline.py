"""把整条流程走一遍:训练几步、对话、评估、诊断。只用极小的假数据和极小的模型,在 CPU 上跑。"""
import csv
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import diagnose
import evaluate
import generate
import train
from common import load_checkpoint, load_run
from prepare_data import BinWriter, encode_record
from tokenizer_tool import load_tokenizer, train_tokenizer

TEXTS = ["今天天气很好,我们一起去公园玩。", "你好,请介绍一下你自己。", "秋天的风吹过山和水,我写了一首诗。"] * 30
TINY_FLOW = [
    "--model", "flow", "--flow_dim", "16", "--flow_time_dim", "16", "--flow_dials", "4", "--flow_heads", "2",
    "--flow_stream_dim", "32", "--flow_context", "32", "--max_outputs", "16", "--attn_rounds", "1", "--review_rounds", "1",
]


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    folder = tmp_path_factory.mktemp("data")
    tokenizer = train_tokenizer(iter(TEXTS), folder / "tokenizer.json", vocab_size=120, limit_alphabet=100)
    for name in ("train", "val"):
        writer = BinWriter(folder / f"{name}.bin")
        for index, text in enumerate(TEXTS):
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            if index % 2:
                writer.write(encode_record([(None, text)], [ids]))
            else:
                writer.write(encode_record([("user", text), ("assistant", text)], [ids[:4], ids]))
        writer.close()
    return folder


@pytest.fixture(autouse=True)
def on_cpu(monkeypatch):
    for module in (train, generate, evaluate, diagnose):
        monkeypatch.setattr(module, "get_device", lambda: torch.device("cpu"))


def run_train(monkeypatch, data_dir, run_dir, *extra):
    arguments = [
        "train.py", "--data_dir", str(data_dir), "--run_dir", str(run_dir), "--max_tokens", "300", "--beat_batch", "4",
        "--max_answer", "12", "--eval_every_tokens", "150", "--eval_beats", "40", "--log_every", "2", *TINY_FLOW, *extra,
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    train.main()


def test_whole_pipeline_runs(monkeypatch, capsys, tmp_path, data_dir):
    run_train(monkeypatch, data_dir, tmp_path)
    run = tmp_path / "flow"
    assert (run / "best.pt").exists() and (run / "last.pt").exists()
    with open(run / "metrics.csv", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) >= 2 and all(float(row["val_loss"]) > 0 for row in rows)
    assert int(rows[-1]["predictions"]) >= 300  # 按实际算过的预测记账
    assert "== 验证" in capsys.readouterr().out
    progress = (run / "进度.txt").read_text(encoding="utf-8")
    assert progress.count("\n    说话损失 ") == len(rows) and "训练结束" in progress

    monkeypatch.setattr(sys, "argv", ["generate.py", "--run", str(run), "--data_dir", str(data_dir), "--prompt", "你好", "--show", "--max_new", "5"])
    generate.main()
    shown = capsys.readouterr().out
    assert "模型:" in shown and "第一拍第 1 路读的比例" in shown

    monkeypatch.setattr(sys, "argv", ["evaluate.py", str(run), "--data_dir", str(data_dir), "--eval_beats", "40"])
    evaluate.main()
    assert "===== 汇总 =====" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["diagnose.py", str(run), "--data_dir", str(data_dir), "--eval_beats", "40", "--batch_size", "4"])
    diagnose.main()
    assert "按词拆分" in capsys.readouterr().out


def test_leak_check_on_saved_model(monkeypatch, tmp_path, data_dir):
    run_train(monkeypatch, data_dir, tmp_path)
    model, cfg, _, _ = load_run(tmp_path / "flow", "cpu", data_dir=data_dir)
    padding_change, inside_change, answer_change, single_gap = diagnose.leak_check(model, cfg.vocab_size, data_dir, "cpu", batch_size=4)
    assert padding_change < 1e-5
    assert inside_change > 0
    assert answer_change == 0
    assert single_gap < 1e-3


def test_resume_keeps_the_saved_model_settings(monkeypatch, tmp_path, data_dir):
    """开关不改变参数的形状。接着训练时命令行漏写开关,模型也不能被悄悄改掉(踩过)。"""
    run_train(monkeypatch, data_dir, tmp_path, "--use_clock", "false")
    first = load_checkpoint(tmp_path / "flow" / "last.pt", "cpu")
    monkeypatch.setattr(sys, "argv", ["train.py", "--data_dir", str(data_dir), "--run_dir", str(tmp_path), "--model", "flow",
                                      "--resume", "true", "--max_tokens", "600", "--beat_batch", "4", "--max_answer", "12",
                                      "--eval_every_tokens", "150", "--eval_beats", "40"])
    train.main()
    payload = load_checkpoint(tmp_path / "flow" / "last.pt", "cpu")
    assert payload["step"] > first["step"] and payload["predictions"] > first["predictions"]
    assert payload["model_cfg"]["use_clock"] is False
    assert payload["model_cfg"]["flow_dim"] == 16


def test_tokenizer_fixture_keeps_special_ids(data_dir):
    assert load_tokenizer(data_dir / "tokenizer.json").token_to_id("<eos>") == 3
