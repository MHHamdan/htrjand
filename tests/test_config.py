"""configs/htrjand_base.yaml is the declared base configuration; this test pins the
code's defaults to it so a silent drift in either place fails the suite."""
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from htrjand import models as M    # noqa: E402
from htrjand import train as T     # noqa: E402

CFG = yaml.safe_load(open(os.path.join(ROOT, "configs", "htrjand_base.yaml")))


def test_every_leaf_carries_a_provenance_tag():
    def walk(node, path=""):
        if isinstance(node, dict) and "value" in node:
            assert "provenance" in node and "reason" in node, path
            assert node["provenance"] in {"PAPER", "FIGURE", "HISTORICAL", "REPOSITORY_PRIOR",
                                          "RECONSTRUCTED_WITH_EVIDENCE", "RECONSTRUCTED_WITH_ASSUMPTION",
                                          "ASSUMPTION", "ASSUMPTION_SELECTED_ON_VALIDATION", "PAPER_AMBIGUITY"}, path
            return
        assert isinstance(node, dict), path
        for k, v in node.items():
            walk(v, f"{path}/{k}")
    walk(CFG)


def test_presets_match_declared_model_values():
    t, s = M.PRESETS["paper_teacher"], M.PRESETS["paper_student"]
    mt, ms = CFG["model"]["teacher"], CFG["model"]["student"]
    assert list(t["channels"]) == mt["channels"]["value"] and len(t["channels"]) == mt["cnn_blocks"]["value"]
    assert [list(p) for p in t["pools"]] == mt["pools"]["value"]
    assert t["lstm_layers"] == mt["lstm_layers"]["value"] and t["lstm_hidden"] == mt["lstm_hidden_per_direction"]["value"]
    assert t["heads"] == mt["heads"]["value"] and t["fusion"] == mt["fusion"]["value"]
    assert t["attn_position"] == mt["attention_position"]["value"] and t["gated"]
    assert list(s["channels"]) == ms["channels"]["value"] and s["se_reduction"] == ms["se_reduction"]["value"]
    assert s["lstm_layers"] == ms["lstm_layers"]["value"] and s["lstm_hidden"] == ms["lstm_hidden_per_direction"]["value"]
    assert s["heads"] == ms["heads"]["value"]


def test_trainer_defaults_match_declared_training_values(monkeypatch, tmp_path):
    captured = {}

    class _Stop(Exception):
        pass

    def fake_build_charset(*a, **k):
        raise _Stop

    monkeypatch.setattr(T.D, "build_charset", fake_build_charset)
    # parse defaults by running main up to the first data access
    import argparse
    orig = argparse.ArgumentParser.parse_args

    def spy(self, argv=None, ns=None):
        a = orig(self, argv, ns)
        captured.update(vars(a))
        return a
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", spy)
    try:
        T.main(["--dataset", "iam", "--out-dir", str(tmp_path), "--tag", "x"])
    except _Stop:
        pass
    tr, kd = CFG["training"], CFG["distillation"]
    assert captured["lr"] == tr["lr"]["value"]
    assert captured["weight_decay"] == tr["weight_decay"]["value"]
    assert captured["batch_size"] == tr["batch_size"]["value"]
    assert captured["epochs"] == tr["epochs"]["value"]
    assert captured["patience"] == 10 and captured["min_delta"] == 1e-3 and captured["es_metric"] == "loss"
    assert captured["clip"] == tr["gradient_clip"]["value"]
    assert captured["aux_weight"] == tr["aux_weight_delta"]["value"] == kd["delta"]["value"]
    assert captured["augment"] is True and captured["normalisation"] == "paper"
    assert list(captured["alpha"]) == kd["alpha"]["value"] and list(captured["gamma"]) == kd["gamma"]["value"]
    assert captured["tau"] == kd["tau"]["value"]
    assert captured["preset"] == "paper_teacher"
