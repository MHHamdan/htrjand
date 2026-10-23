"""Tests of the release additions (hub format, preprocessing, decoders, CLI). CPU only,
no archives, no network:  python -m pytest tests -q"""
import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
cv2 = pytest.importorskip("cv2")

from htrjand import HTRJANDModel, NotALineError  # noqa: E402
from htrjand import data as D  # noqa: E402
from htrjand import decode as Dec  # noqa: E402
from htrjand import inference as I  # noqa: E402
from htrjand import models as M  # noqa: E402
from htrjand.hub import save_pretrained, sha256_file  # noqa: E402

# the release charset (training transcriptions of the five corpora), 108 characters -> V = 111
CHARSET = (" !\"#%&'()*+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_abcdefghijklmnopqrstuvwxyz{|}"
           "£¤§°²ÀÉàâçèéêëîôùûœ€⊥")
STUDENT = dict(input_size=(128, 1632), channels=(16, 32, 64), pools=((2, 2), (2, 2), (4, 2)), gated=True, se=True,
               se_reduction=8, lstm_hidden=64, lstm_layers=4, heads=1, attn="combined", attn_position="after_lstm",
               fusion="paper", dropout=0.05, aux_head=True)
TEACHER = dict(input_size=(128, 1632), channels=(32, 64, 128, 256, 512), pools=((2, 2), (2, 2), (2, 2), (2, 1), (2, 1)),
               gated=True, se=True, se_reduction=16, lstm_hidden=128, lstm_layers=4, heads=2, attn="combined",
               attn_position="after_lstm", fusion="paper", dropout=0.2, dropout_cnn=0.1, dropout_rnn=0.2, aux_head=True)
TINY = dict(input_size=(32, 256), channels=(4, 8), pools=((2, 2), (4, 2)), gated=True, se=True, se_reduction=2,
            lstm_hidden=8, lstm_layers=1, heads=1, aux_head=True)


def tiny_model(charset="abcdefghijklmnopqrstuvwxyz ", seed=0):
    torch.manual_seed(seed)
    return M.HTRJAND(D.Tokenizer(charset).vocab_size, **TINY).eval()


def render_line(text="hello world", scale=1.2, thickness=2, pad=(30, 40), dark_bg=False):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, scale, thickness)
    img = np.full((h + base + 2 * pad[0], w + 2 * pad[1]), 255, np.uint8)
    cv2.putText(img, text, (pad[1], pad[0] + h), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, scale, 0, thickness, cv2.LINE_AA)
    return 255 - img if dark_bg else img


# ------------------------------------------------------------------ released architectures
def test_release_charset_and_parameter_counts():
    tok = D.Tokenizer(CHARSET)
    assert len(tok.charset) == 108 and tok.vocab_size == 111 and tok.BLANK == 110
    s, t = M.HTRJAND(111, **STUDENT), M.HTRJAND(111, **TEACHER)
    assert s.deployed_parameters() == 806_287 and M.count_params(s) == 863_230
    assert t.deployed_parameters() == 7_220_911 and M.count_params(t) == 7_448_350
    assert s.n_frames == t.n_frames == 204


# ------------------------------------------------------------------ release format
def test_save_and_from_pretrained_roundtrip(tmp_path):
    net = tiny_model()
    root = tmp_path / "repo"
    save_pretrained(net, "abcdefghijklmnopqrstuvwxyz ", str(root / "tiny"))
    cfg = json.load(open(root / "tiny" / "config.json"))
    assert cfg["architecture"] == json.loads(json.dumps(net.cfg))           # config round-trip
    assert cfg["weights"]["sha256"] == sha256_file(str(root / "tiny" / "model.safetensors"))
    m = HTRJANDModel.from_pretrained(str(root), variant="tiny", device="cpu")   # repo root + variant
    m2 = HTRJANDModel.from_pretrained(str(root / "tiny"), device="cpu")       # variant dir directly
    x = torch.randn(2, 1, 32, 256)
    with torch.no_grad():
        ref = net(x)
    assert torch.equal(m.logits(x), ref) and torch.equal(m2.logits(x), ref)
    assert m.deployed_parameters() == net.deployed_parameters()
    with pytest.raises(FileNotFoundError):
        HTRJANDModel.from_pretrained(str(root), variant="nope")


def test_from_pretrained_is_strict_and_checks_the_hash(tmp_path):
    d = tmp_path / "v"
    save_pretrained(tiny_model(), "abcdefghijklmnopqrstuvwxyz ", str(d))
    cfg = json.load(open(d / "config.json"))
    cfg["weights"]["sha256"] = "0" * 64
    json.dump(cfg, open(d / "config.json", "w"))
    with pytest.raises(ValueError, match="SHA-256"):
        HTRJANDModel.from_pretrained(str(d))
    cfg["weights"]["sha256"] = sha256_file(str(d / "model.safetensors"))
    cfg["architecture"]["lstm_hidden"] = 9                                       # shapes no longer match
    json.dump(cfg, open(d / "config.json", "w"))
    with pytest.raises(RuntimeError):
        HTRJANDModel.from_pretrained(str(d))


# ------------------------------------------------------------------ preprocessing
def ink_box(canvas_bright_ink):
    ys, xs = np.where(canvas_bright_ink > 128)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


@pytest.mark.parametrize("h,w", [(40, 400), (60, 1500), (200, 900), (30, 3000)])
def test_fit_canvas_keeps_aspect_ratio_and_inverts(h, w):
    img = np.full((h, w), 255, np.uint8)                                # white page, dark frame at the edges
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), 0, 2)
    c = I.fit_canvas(img, (128, 1632))
    assert c.shape == (128, 1632) and c.dtype == np.uint8
    assert c.mean() < 128                                               # ink bright on a dark ground
    t, b, l, r = ink_box(c)
    s = min(1632 / w, 128 / h)
    assert abs((b - t) - int(h * s)) <= 1 and abs((r - l) - int(w * s)) <= 1   # one scale factor, both axes
    assert abs((t + b) / 2 - 64) <= 1 and abs((l + r) / 2 - 816) <= 1           # centre padding


def test_fit_canvas_matches_the_archive_builder(tmp_path):
    from htrjand.prepare_data import preprocess
    img = render_line("archive recipe", 1.0)
    p = str(tmp_path / "l.png")
    cv2.imwrite(p, img)
    assert np.array_equal(preprocess(p, (128, 1632)), I.fit_canvas(img, (128, 1632)))
    r = I.prepare_line(p, (128, 1632), crop=False, check=False)
    assert np.array_equal(r.canvas, preprocess(p, (128, 1632)))
    x = I.normalise_canvas(r.canvas)
    assert x.min() == -1.0 and x.max() == 1.0                           # per-image min-max normalisation


def test_dark_background_and_rgba_inputs():
    light = I.prepare_line(render_line(), (128, 1632))
    dark = I.prepare_line(render_line(dark_bg=True), (128, 1632))
    assert not light.inverted_input and dark.inverted_input
    assert np.array_equal(light.canvas, dark.canvas)
    g = render_line()
    rgba = np.zeros(g.shape + (4,), np.uint8)
    rgba[..., :3] = 0
    rgba[..., 3] = 255 - g                                              # ink opaque, background transparent
    r = I.prepare_line(rgba, (128, 1632))
    assert not r.inverted_input and r.canvas.mean() < 128
    t, b, l, rr = ink_box(r.canvas)
    t2, b2, l2, r2 = ink_box(light.canvas)
    assert abs((rr - l) - (r2 - l2)) <= 4


def test_whitespace_crop():
    g = render_line(pad=(200, 600))
    raw = I.prepare_line(g, (128, 1632), crop=False)
    cropped = I.prepare_line(g, (128, 1632))
    _, _, l0, r0 = ink_box(raw.canvas)
    _, _, l1, r1 = ink_box(cropped.canvas)
    assert (r1 - l1) > 2 * (r0 - l0)                                    # the writing fills the canvas
    h, w = cropped.cropped.shape
    assert h < g.shape[0] / 3 and w < g.shape[1]


def test_page_and_multiline_detection(tmp_path):
    a, b = render_line("first line here"), render_line("second line too")
    w = max(a.shape[1], b.shape[1])
    pad = lambda x: np.pad(x, ((0, 0), (0, w - x.shape[1])), constant_values=255)
    two = np.vstack([pad(a), np.full((40, w), 255, np.uint8), pad(b)])
    ok, why = I.line_check(I.crop_to_ink(two)[0])
    assert not ok and "bands" in why
    assert I.line_check(I.crop_to_ink(render_line())[0])[0]
    tall = np.full((600, 200), 255, np.uint8)
    cv2.putText(tall, "a", (60, 300), cv2.FONT_HERSHEY_SIMPLEX, 8, 0, 20)
    assert not I.prepare_line(tall, (128, 1632)).is_line
    d = tmp_path / "m"
    save_pretrained(tiny_model(), "abcdefghijklmnopqrstuvwxyz ", str(d))
    m = HTRJANDModel.from_pretrained(str(d), device="cpu")
    with pytest.raises(NotALineError):
        m.transcribe(two)
    r = m.transcribe(two, force=True)
    assert not r.is_line and r.warnings
    from htrjand.segment_lines import split_lines
    parts = split_lines(two)
    assert len(parts) == 2
    assert len(m.transcribe(two, segment_lines=True)) == 2


def test_small_angle_deskew():
    g = render_line("deskew this line of text", 1.0, pad=(60, 60))
    for ang in (-3.0, 2.0):
        rot = I.rotate(g, ang)
        est = I.estimate_skew(I.crop_to_ink(rot)[0])
        assert abs(est + ang) <= 0.75, (ang, est)                     # estimate undoes the rotation
    assert abs(I.estimate_skew(I.crop_to_ink(g)[0])) <= 0.5


# ------------------------------------------------------------------ decoders, confidence
def onehot_logprobs(seq, chars, T=None, blank=None, p=0.9):
    """(T, V) log-probs spelling `seq` (list of indices, None = blank) with confidence p."""
    V = len(chars) + 1
    blank = V - 1 if blank is None else blank
    lp = np.full((len(seq), V), np.log((1 - p) / (V - 1)), np.float32)
    for t, k in enumerate(seq):
        lp[t, blank if k is None else k] = np.log(p)
    return lp


def test_ctc_greedy_and_confidence():
    tok = D.Tokenizer("ab")
    a, b = tok.char_to_index["a"], tok.char_to_index["b"]
    lp = onehot_logprobs([a, a, None, a, b, b, None], tok.chars)
    assert tok.ctc_greedy_decode(torch.from_numpy(lp[None]))[0] == "aab"
    probs = np.exp(lp)
    assert abs(I.frame_confidence(probs, tok.BLANK) - 0.9) < 1e-6
    assert I.char_confidences(probs, tok.BLANK) == pytest.approx([0.9, 0.9, 0.9])


def test_compact_charlm_roundtrip_equivalence_and_pruning(tmp_path):
    txt = ["the quick brown fox jumps over the lazy dog", "the dog sleeps", "a quick test of the model"] * 3
    full, comp = Dec.CharLM(6).fit(txt), Dec.CompactCharLM(6).fit(txt)
    path = comp.save(str(tmp_path / "lm.gz"), lang="en")
    back = Dec.load_lm(path)
    assert back.ngrams == comp.ngrams and back.totals == comp.totals and back.V == comp.V
    rng = random.Random(0)
    chars = sorted(set("".join(txt))) + ["Z"]
    for _ in range(3000):
        s = rng.choice(txt)
        i = rng.randrange(len(s))
        ctx = s[max(0, i - rng.randint(0, 8)):i] if rng.random() < 0.7 else "".join(rng.choice(chars) for _ in range(rng.randint(0, 6)))
        ch = s[i] if rng.random() < 0.7 else rng.choice(chars)
        assert abs(full.logp(ctx, ch) - back.logp(ctx, ch)) < 1e-12
    pruned = Dec.CompactCharLM(6).fit(txt, min_count=4)
    assert pruned.n_entries() < comp.n_entries()
    assert pruned.logp("the", " ") == comp.logp("the", " ")               # frequent n-gram kept exactly


def test_lm_rescoring_follows_the_language_model():
    from htrjand.lmdecode import nbest_rescore
    tok = D.Tokenizer("abcdefghijklmnopqrstuvwxyz ")
    ix = tok.char_to_index
    seq = [ix["c"], None, ix["a"], None, "amb", None]
    V = tok.vocab_size
    lp = np.full((len(seq), V), np.log(1e-4), np.float32)
    for t, k in enumerate(seq):
        if k == "amb":                                                  # "t" vs "r", acoustically tied
            lp[t, ix["t"]], lp[t, ix["r"]] = np.log(0.45), np.log(0.45)
        else:
            lp[t, tok.BLANK if k is None else k] = np.log(0.99)
    lp = lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)
    lm = Dec.CompactCharLM(4).fit(["the cat sat", "a cat", "cat"] * 5)
    assert nbest_rescore(lp, tok, lm, alpha=1.0, beta=0.0) == "cat"
    lm2 = Dec.CompactCharLM(4).fit(["the car", "a car", "car"] * 5)
    assert nbest_rescore(lp, tok, lm2, alpha=1.0, beta=0.0) == "car"


def test_transcribe_with_lm_decoder(tmp_path, monkeypatch):
    d = tmp_path / "m"
    cs = "abcdefghijklmnopqrstuvwxyz "
    save_pretrained(tiny_model(cs), cs, str(d), extra=None)
    cfg = json.load(open(d / "config.json"))
    cfg["decoders"]["lm"]["per_corpus"] = {"iam": dict(lm_file="charlm_en.gz", alpha=0.3, beta=1.0)}
    json.dump(cfg, open(d / "config.json", "w"))
    lmdir = tmp_path / "lm"
    lmdir.mkdir()
    Dec.CompactCharLM(5).fit(["hello world", "the quick brown fox"]).save(str(lmdir / "charlm_en.gz"))
    monkeypatch.setenv("HTRJAND_CHARLM_DIR", str(lmdir))
    m = HTRJANDModel.from_pretrained(str(d), device="cpu")
    r = m.transcribe(render_line(), decoder="lm")
    assert isinstance(r.lm, str) and r.text == r.lm and 0.0 <= r.confidence <= 1.0


# ------------------------------------------------------------------ CLI and scripts
def run(*args, env=None):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=300,
                          env=dict(os.environ, **(env or {})))


def example_image(tmp_path):
    ex = sorted(f for f in os.listdir(os.path.join(ROOT, "examples")) if f.endswith(".png")) \
        if os.path.isdir(os.path.join(ROOT, "examples")) else []
    if ex:
        return os.path.join(ROOT, "examples", ex[0])
    p = str(tmp_path / "line.png")
    cv2.imwrite(p, render_line())
    return p


def test_predict_cli_smoke(tmp_path):
    save_pretrained(tiny_model(), "abcdefghijklmnopqrstuvwxyz ", str(tmp_path / "repo" / "tiny"))
    r = run("predict.py", "--model", str(tmp_path / "repo"), "--variant", "tiny", "--image", example_image(tmp_path),
            "--device", "cpu", "--force", "--save-preprocessed", str(tmp_path / "prep"))
    assert r.returncode == 0, r.stderr
    assert "raw CTC" in r.stdout and "confidence" in r.stdout
    assert os.listdir(tmp_path / "prep")
    j = run("predict.py", "--model", str(tmp_path / "repo" / "tiny"), "--image", example_image(tmp_path), "--json", "--force")
    assert j.returncode == 0 and json.loads(j.stdout.splitlines()[0])["decoder"] == "greedy"


@pytest.mark.parametrize("cmd", [["predict.py", "--help"], ["-m", "htrjand.train", "--help"],
                                 ["-m", "htrjand.evaluate", "--help"], ["scripts/registry.py", "--help"],
                                 ["scripts/leakage_audit.py", "--help"], ["scripts/final_table.py", "--help"],
                                 ["scripts/export_hf.py", "--help"], ["scripts/build_release_lm.py", "--help"],
                                 ["scripts/eval_release_decoders.py", "--help"], ["scripts/verify_export.py", "--help"],
                                 ["scripts/build_ood_figure.py", "--help"], ["scripts/build_corpus_lm.py", "--help"],
                                 ["scripts/release_hygiene.py", "--help"]])
def test_command_line_help(cmd):
    r = run(*cmd)
    assert r.returncode == 0 and "usage" in r.stdout.lower(), r.stderr


def test_registry_fails_clearly_without_runs(tmp_path):
    r = run("scripts/registry.py", "--runs-dir", str(tmp_path / "none"), "--out-dir", str(tmp_path / "o"))
    assert r.returncode != 0 and "not part of the public" in r.stderr


def test_final_table_reruns_from_the_shipped_results(tmp_path):
    r = run("scripts/final_table.py", "--out-dir", str(tmp_path))
    assert r.returncode == 0 and (tmp_path / "final_selection.json").exists()


def test_import_needs_no_training_or_demo_dependencies():
    code = ("import sys\n"
            "for m in ('h5py', 'transformers', 'gradio', 'editdistance', 'yaml'): sys.modules[m] = None\n"
            "import htrjand\n"
            "from htrjand import HTRJANDModel, prepare_line\n"
            "import htrjand.hub, htrjand.inference, htrjand.lmdecode, htrjand.segment_lines, htrjand.decode\n"
            "print('ok')\n")
    r = run("-c", code)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


def test_hub_id_path_uses_variant_allow_patterns(tmp_path, monkeypatch):
    """The Hub branch of from_pretrained (no network: snapshot_download is replaced)."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    save_pretrained(tiny_model(), "abcdefghijklmnopqrstuvwxyz ", str(tmp_path / "snap" / "iam"))
    seen = {}

    def fake_snapshot_download(repo_id, revision=None, allow_patterns=None, **kw):
        seen.update(repo_id=repo_id, revision=revision, allow_patterns=allow_patterns)
        return str(tmp_path / "snap")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    m = HTRJANDModel.from_pretrained("someone/some-repo", variant="iam", revision="v1.0", device="cpu")
    assert seen["repo_id"] == "someone/some-repo" and seen["revision"] == "v1.0"
    assert "iam/*" in seen["allow_patterns"] and not any(p.startswith("general") for p in seen["allow_patterns"])
    assert m.hub_id == "someone/some-repo"


def test_ood_figure_script_imports():
    pytest.importorskip("matplotlib")
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_ood_figure", os.path.join(ROOT, "scripts", "build_ood_figure.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(mod.main) and mod.readings()                  # parses examples/SOURCES.md
    assert set(mod.readings()) <= set(os.listdir(os.path.join(ROOT, "examples")))


def test_named_release_presets_equal_the_released_networks():
    s = M.build("htrjand_student", 111, input_size=(128, 1632))
    t = M.build("htrjand_teacher", 111, input_size=(128, 1632))
    assert s.deployed_parameters() == 806_287 and s.n_frames == 204 and s.column_dim == 512
    assert t.deployed_parameters() == 7_220_911 and t.n_frames == 204
    q = M.build("paper_student", 111, input_size=(128, 1632), pools=[[2, 2], [2, 2], [4, 2]])
    assert q.cfg == s.cfg


def test_fetch_synth_fonts_from_google_fonts(tmp_path):
    """Network test: three fonts of configs/synth_fonts.json download from the pinned google/fonts commit and
    match their SHA-256 (skipped without network)."""
    import socket
    try:
        socket.create_connection(("raw.githubusercontent.com", 443), timeout=5).close()
    except OSError:
        pytest.skip("no network")
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import fetch_synth_resources as F
    path, n = F.fetch_fonts(str(tmp_path), limit=3)
    cov = json.load(open(path))
    assert n == 3 and len(cov) == 3 and all(os.path.exists(k) for k in cov)


def test_tiny_images_are_refused_with_a_clear_message():
    with pytest.raises(ValueError, match="too small"):
        I.prepare_line(np.full((10, 400), 255, np.uint8))
    with pytest.raises(ValueError, match="too small"):
        I.prepare_line(np.full((1, 1), 0, np.uint8))
    r = I.prepare_line(cv2.resize(render_line("tiny line"), (120, 18)))
    assert any("low resolution" in w for w in r.warnings)


def test_segment_lines_never_raises_on_a_page(tmp_path):
    lines = [render_line(t) for t in ("first line here", "second line", "third one", "fourth line of text",
                                      "fifth", "sixth line")]
    w = max(x.shape[1] for x in lines)
    page = np.vstack([np.vstack([np.pad(x, ((0, 0), (0, w - x.shape[1])), constant_values=255),
                                 np.full((20, w), 255, np.uint8)]) for x in lines])
    d = tmp_path / "m"
    save_pretrained(tiny_model(), "abcdefghijklmnopqrstuvwxyz ", str(d))
    m = HTRJANDModel.from_pretrained(str(d), device="cpu")
    out = m.transcribe(page, segment_lines=True)
    assert len(out) >= 5


def test_white_ink_on_transparent_background():
    g = render_line()
    rgba = np.zeros(g.shape + (4,), np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = 255 - g                                   # white strokes drawn as opacity
    r = I.prepare_line(rgba, (128, 1632))
    assert not any("no ink" in w for w in r.warnings) and r.canvas.max() > 128


def test_app_handles_tiny_images_without_crashing(tmp_path, monkeypatch):
    pytest.importorskip("gradio")
    save_pretrained(tiny_model(), "abcdefghijklmnopqrstuvwxyz ", str(tmp_path / "htrjand-student" / "general"))
    monkeypatch.setenv("HTRJAND_MODEL_ROOT", str(tmp_path))
    import importlib.util
    spec = importlib.util.spec_from_file_location("htrjand_app", os.path.join(ROOT, "app.py"))
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    from PIL import Image
    out = app.run(Image.new("L", (1, 1), 0), next(iter(app.MODELS)), "English (IAM)", app.DECODERS[0], False, False, False)
    assert "too small" in out[-1]
