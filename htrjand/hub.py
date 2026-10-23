"""Released-model API: load a model from a Hugging Face repo or a local directory,
transcribe line images (release addition).

    from htrjand import HTRJANDModel
    model = HTRJANDModel.from_pretrained("MHamdan/htrjand-student", variant="general")
    r = model.transcribe("line.png", decoder="greedy", return_confidence=True)
    print(r.text, r.confidence)

Release layout (one repo, one sub-folder per variant):

    <repo>/README.md                     model card
    <repo>/<variant>/config.json         everything needed to rebuild the network
    <repo>/<variant>/model.safetensors   weights (strict load)

The released character LMs and the T5 lexicons are Wikipedia-derived (CC BY-SA 4.0) and live in
their own repo (release.CHARLM, `charlm_<lang>.gz`, `lexicon/wiki_<lang>.txt.gz`); the T5 corrector
is release.T5_CORRECTOR. A local model directory finds both as sibling directories.

`from_pretrained` builds the network from config.json alone (no preset table, no
tensor-shape inference) and loads the weights with `strict=True`; the SHA-256 of
the weights file is checked against the config.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os

import numpy as np
import torch

from . import data as D
from . import decode as Dec
from .inference import (NotALineError, char_confidences, frame_confidence, prepare_line)
from .models import HTRJAND

from .release import DEFAULT_CHARLM, DEFAULT_MODEL as DEFAULT_REPO, DEFAULT_T5 as DEFAULT_T5_REPO  # noqa: E402

CONFIG_FORMAT = "htrjand/1"
LANG_OF_CORPUS = dict(iam="en", rimes="fr", bentham="en", saintgall="la", washington="en")

_LM_CACHE = {}


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _is_hub_id(name):
    return not os.path.exists(name) and name.count("/") == 1 and not name.startswith((".", "/", "~"))


def resolve(name_or_path, variant=None, revision=None, cache_dir=None, token=None,
            local_files_only=False, extra_patterns=()):
    """Return (repo_root, variant_dir). `name_or_path` is a local directory (the repo
    root holding <variant>/, or a variant directory itself) or a Hub repo id."""
    if not _is_hub_id(name_or_path):
        root = os.path.abspath(os.path.expanduser(name_or_path))
        if not os.path.isdir(root):
            raise FileNotFoundError(f"{name_or_path}: no such directory")
        if variant and os.path.exists(os.path.join(root, variant, "config.json")):
            return root, os.path.join(root, variant)
        if os.path.exists(os.path.join(root, "config.json")):
            return os.path.dirname(root), root
        found = sorted(d for d in os.listdir(root) if os.path.exists(os.path.join(root, d, "config.json")))
        raise FileNotFoundError(f"{root}: variant {variant!r} not found; available: {found}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError("loading from the Hugging Face Hub needs `pip install huggingface_hub`") from e
    variant = variant or "general"
    root = snapshot_download(repo_id=name_or_path, revision=revision, cache_dir=cache_dir, token=token,
                             local_files_only=local_files_only,
                             allow_patterns=[f"{variant}/*", "README.md", *extra_patterns])
    vdir = os.path.join(root, variant)
    if not os.path.exists(os.path.join(vdir, "config.json")):
        raise FileNotFoundError(f"{name_or_path}@{revision or 'main'}: no variant {variant!r}")
    return root, vdir


def fetch(repo_root, rel, hub_id=None, revision=None, cache_dir=None, token=None):
    """A file of the repo: local if present, else downloaded from the Hub."""
    p = os.path.join(repo_root, rel)
    if os.path.exists(p):
        return p
    if hub_id is None:
        raise FileNotFoundError(p)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=hub_id, filename=rel, revision=revision, cache_dir=cache_dir, token=token)


@dataclass
class Transcription:
    text: str                                   # final output (T5 > LM > raw, whichever was run)
    raw: str                                    # greedy CTC
    lm: str = None                              # character-LM beam search + rescoring
    t5: str = None                              # T5 refinement (gated)
    decoder: str = "greedy"
    confidence: float = None                    # mean max-softmax over non-blank greedy frames
    char_confidences: list = None               # per emitted character of the greedy path
    eps_hat: float = None                       # 1 - mean_t max_c p(c|x_t), the T5 gate's error estimate
    t5_confidence: float = None                 # exp(length-normalised beam log-prob) of the T5 candidate
    t5_applied: bool = None
    is_line: bool = True
    warnings: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class HTRJANDModel:
    """A released HTR-JAND recogniser (Student or Teacher) with its decoders."""

    def __init__(self, net, config, repo_root=None, variant_dir=None, hub_id=None, revision=None,
                 device="cpu", cache_dir=None, token=None):
        self.net = net.eval()
        self.config = config
        self.repo_root, self.variant_dir = repo_root, variant_dir
        self.hub_id, self.revision, self.cache_dir, self.token = hub_id, revision, cache_dir, token
        tk = config["tokenizer"]
        self.tokenizer = D.Tokenizer(tk["charset"])
        if self.tokenizer.chars != tk["chars"] or self.tokenizer.BLANK != tk["blank_index"]:
            raise ValueError("config tokenizer does not match the charset")
        self.input_size = tuple(config["preprocessing"]["input_size"])
        self.normalisation = config["preprocessing"]["normalisation"]
        self.device = torch.device(device)
        self.net.to(self.device)
        self._corrector = None

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, name_or_path=DEFAULT_REPO, variant="general", revision=None, device=None,
                        cache_dir=None, token=None, local_files_only=False, verify=True):
        root, vdir = resolve(name_or_path, variant, revision, cache_dir, token, local_files_only)
        config = json.load(open(os.path.join(vdir, "config.json"), encoding="utf-8"))
        if config.get("format") != CONFIG_FORMAT:
            raise ValueError(f"{vdir}/config.json: unsupported format {config.get('format')!r}")
        wfile = os.path.join(vdir, config["weights"]["file"])
        if verify and sha256_file(wfile) != config["weights"]["sha256"]:
            raise ValueError(f"{wfile}: SHA-256 differs from config.json")
        net = build_from_config(config)
        from safetensors.torch import load_file
        net.load_state_dict(load_file(wfile), strict=True)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls(net, config, root, vdir, name_or_path if _is_hub_id(name_or_path) else None,
                   revision, device, cache_dir, token)

    @property
    def corpus(self):
        return self.config["corpus"]["default"]

    def deployed_parameters(self):
        return self.net.deployed_parameters()

    # ------------------------------------------------------------------ core
    def preprocess(self, image, crop=True, deskew=False):
        return prepare_line(image, self.input_size, self.normalisation, crop=crop, deskew=deskew)

    @torch.no_grad()
    def log_probs(self, x):
        """x: float32 (B, 1, H, W) normalised canvases -> (B, T, V) log-softmax, numpy."""
        x = torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)
        return torch.log_softmax(self.net(x).float(), -1).cpu().numpy()

    @torch.no_grad()
    def logits(self, x):
        x = torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)
        return self.net(x).float().cpu()

    def greedy(self, lp):
        return self.tokenizer.ctc_greedy_decode(torch.from_numpy(np.asarray(lp)))

    # ------------------------------------------------------------------ LM
    def lm_settings(self, corpus=None):
        corpus = corpus or self.corpus
        s = self.config["decoders"]["lm"]["per_corpus"].get(corpus)
        if s is None:
            raise KeyError(f"no LM settings for corpus {corpus!r}; "
                           f"available: {sorted(self.config['decoders']['lm']['per_corpus'])}")
        return dict(self.config["decoders"]["lm"]["search"], **s)

    def companion(self, repo):
        """(local directory, None) for a companion repo (LM, corrector) staged next to this
        model's local directory, else (None, hub id)."""
        name = repo.rstrip("/").split("/")[-1]
        if self.hub_id is None and self.repo_root:
            cand = os.path.join(os.path.dirname(self.repo_root), name)
            if os.path.isdir(cand):
                return cand, None
        return None, repo

    def lm_path(self, corpus=None):
        """Released LM file for `corpus`: $HTRJAND_CHARLM_DIR, a sibling directory of a local
        model, or the LM repo on the Hub (config.json decoders.lm.repo)."""
        s = self.lm_settings(corpus)
        env = os.environ.get("HTRJAND_CHARLM_DIR")
        if env:
            return os.path.join(env, s["lm_file"])
        local, hub = self.companion(self.config["decoders"]["lm"]["repo"])
        if local:
            return os.path.join(local, s["lm_file"])
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=hub, filename=s["lm_file"], cache_dir=self.cache_dir, token=self.token)

    def load_lm(self, corpus=None, path=None):
        if path is None:
            path = self.lm_path(corpus)
        key = os.path.abspath(path)
        if key not in _LM_CACHE:
            _LM_CACHE[key] = Dec.load_lm(path)
        return _LM_CACHE[key]

    def decode_lm(self, lp, corpus=None, lm=None, settings=None):
        """Character-LM decoding of one line, the single-model case of
        `ensemble_rescore.py`: CTC prefix beam search with the LM (gen_alpha, gen_beta)
        returns the n-best list; each candidate is rescored as
        log p_CTC(c|x) + alpha * log P_LM(c) + beta * |c| and the best is returned."""
        from .lmdecode import nbest_rescore
        s = settings or self.lm_settings(corpus)
        lm = lm or self.load_lm(corpus)
        return nbest_rescore(np.asarray(lp), self.tokenizer, lm, beam=s["beam"], nbest=s["nbest"],
                             gen_alpha=s["gen_alpha"], gen_beta=s["gen_beta"], alpha=s["alpha"],
                             beta=s["beta"], prune=s.get("prune", -9.0))

    # ------------------------------------------------------------------ T5
    def load_corrector(self, name_or_path=None, device=None):
        from .t5correct import T5Corrector
        if name_or_path is None:
            local, hub = self.companion(self.config["postprocessing"]["t5"]["corrector"])
            name_or_path = local or hub
        lex_local, lex_hub = self.companion(self.config["decoders"]["lm"]["repo"])
        self._corrector = T5Corrector.from_pretrained(name_or_path, device=device or "cpu",
                                                      resources=os.environ.get("HTRJAND_CHARLM_DIR") or lex_local or lex_hub,
                                                      cache_dir=self.cache_dir, token=self.token)
        return self._corrector

    def t5_gate(self, corpus=None, base="ctc"):
        corpus = corpus or self.corpus
        g = self.config["postprocessing"]["t5"]["gates"].get(corpus, {}).get(base)
        if g is None:
            raise KeyError(f"no T5 gate for corpus {corpus!r}, base {base!r}")
        return g

    # ------------------------------------------------------------------ public
    def transcribe(self, images, decoder="greedy", return_confidence=True, corpus=None, t5=False,
                   force=False, crop=True, deskew=False, segment_lines=False, lines_in_order=None,
                   batch_size=16):
        """Transcribe one line image (path, PIL image or array) or a list of them.

        decoder        "greedy" (raw CTC) or "lm" (released Wikipedia character LM)
        corpus         domain whose LM language / T5 gate is used (default: the variant's)
        t5             also run the gated T5 corrector on the chosen decoder's output
        force          transcribe inputs that look like a page / several lines anyway
        segment_lines  split each input into lines with the naive external helper first
                       (htrjand.segment_lines; not part of the evaluated system)
        Returns a Transcription, or a list for a list input / segment_lines=True."""
        if decoder not in ("greedy", "lm"):
            raise ValueError("decoder must be 'greedy' or 'lm'")
        single = not isinstance(images, (list, tuple))
        items = [images] if single else list(images)
        if segment_lines:
            from .segment_lines import split_lines
            items = [crop_ for im in items for crop_ in split_lines(im)]
            single = False
            lines_in_order = True if lines_in_order is None else lines_in_order
        prepared = []
        for im in items:
            p = self.preprocess(im, crop=crop, deskew=deskew)
            if not p.is_line and not force:
                raise NotALineError(
                    "; ".join(p.warnings) + ". HTR-JAND is a line-level recogniser: crop one text line, "
                    "pass force=True (CLI: --force) to transcribe anyway, or segment_lines=True "
                    "(CLI: --segment-lines) to split the image with a naive external line splitter.")
            prepared.append(p)
        out = []
        for i in range(0, len(prepared), batch_size):
            chunk = prepared[i:i + batch_size]
            lp = self.log_probs(np.stack([p.tensor_input for p in chunk]))
            raws = self.greedy(lp)
            for p, l, raw in zip(chunk, lp, raws):
                probs = np.exp(l)
                r = Transcription(text=raw, raw=raw, decoder=decoder, is_line=p.is_line, warnings=list(p.warnings))
                if return_confidence:
                    r.confidence = frame_confidence(probs, self.tokenizer.BLANK)
                    r.char_confidences = char_confidences(probs, self.tokenizer.BLANK, self.tokenizer.PAD)
                r.eps_hat = float(1.0 - probs.max(-1).mean())
                if decoder == "lm":
                    r.lm = self.decode_lm(l, corpus)
                    r.text = r.lm
                out.append(r)
        if t5:
            corr = self._corrector or self.load_corrector()
            base = "lm" if decoder == "lm" else "ctc"
            gate = self.t5_gate(corpus, base)
            hyps = [r.text for r in out]
            eps = [r.eps_hat if base == "ctc" else float("nan") for r in out]
            res = corr.correct(hyps, corpus or self.corpus, gate, eps=eps,
                               context=bool(lines_in_order))
            for r, (txt, conf, applied) in zip(out, res):
                r.t5, r.t5_confidence, r.t5_applied = txt, conf, applied
                r.text = txt
        return out[0] if single else out


def build_from_config(config):
    a = dict(config["architecture"])
    return HTRJAND(**a)


def save_pretrained(net, charset, out_dir, normalisation="paper", variant=None, extra=None):
    """Write a minimal release-format directory (config.json + model.safetensors) for a
    network of this package, e.g. after fine-tuning. `scripts/export_hf.py` writes the
    full configs of the released models (provenance, metrics, decoder settings)."""
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    tok = D.Tokenizer(charset)
    if tok.vocab_size != net.cfg["vocab_size"]:
        raise ValueError("charset does not match the network's output layer")
    wpath = os.path.join(out_dir, "model.safetensors")
    save_file({k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()}, wpath)
    n_aux = sum(p.numel() for p in net.aux.parameters()) if net.aux is not None else 0
    n = sum(p.numel() for p in net.parameters())
    cfg = dict(format=CONFIG_FORMAT, model_type="htrjand", variant=variant or os.path.basename(out_dir.rstrip("/")),
               role="custom", architecture=dict(net.cfg),
               parameters=dict(total=n, deployed=n - n_aux, aux_head=n_aux),
               preprocessing=dict(input_size=list(net.cfg["input_size"]), channels=1, normalisation=normalisation),
               tokenizer=dict(charset=tok.charset, chars=tok.chars, pad_index=tok.PAD, unk_index=tok.UNK,
                              blank_index=tok.BLANK, vocab_size=tok.vocab_size),
               corpus=dict(default="iam", supported=[]),
               decoders=dict(default="greedy", lm=dict(repo=DEFAULT_CHARLM, search=dict(beam=16, nbest=8, prune=-9.0,
                                                                                    gen_alpha=0.2, gen_beta=0.5),
                                                   per_corpus={})),
               postprocessing=dict(t5=dict(corrector=DEFAULT_T5_REPO, gates={})),
               weights=dict(file="model.safetensors", sha256=sha256_file(wpath), bytes=os.path.getsize(wpath),
                            includes_aux_head=n_aux > 0))
    if extra:
        cfg.update(extra)
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return out_dir
