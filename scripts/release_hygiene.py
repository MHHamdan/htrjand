#!/usr/bin/env python3
"""Release hygiene gate for the public repository (and, optionally, the Hugging Face staging dir).

    python scripts/release_hygiene.py [--root .] [--hf-root ../htrjand-release-hf] [--data-dir $HTRJAND_DATA_DIR]

Fails (exit 1) on: a file > 5 MB; weights / archives / caches (*.pth, *.pt, *.ckpt, *.hdf5, *.safetensors,
*.npy, *.npz); per-line prediction or T5 cache files; machine paths (the author's home directory, agent scratch directories);
the training host name; references to the earlier pre-print (the archive name, its identifier, the word itself)
outside EARLIER_REF_ALLOW; anything that looks like an access token (hf_..., ghp_..., gho_..., github_pat_...);
notebook outputs; image files outside examples/ (SOURCES.md), docs/figures/ (FIGURES.md) or docs/assets/ (build_ood_figure.py), and the paper-only qualitative.* figure; and, with --data-dir,
any benchmark transcription line (>= 25 characters) found verbatim in a text file. With --hf-root the
staged repos are scanned for paths, host names and tokens (weights there are expected).
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = (".pth", ".pt", ".ckpt", ".hdf5", ".h5", ".safetensors", ".npy", ".npz", ".whl", ".tar.gz", ".zip")
IMAGES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".tif", ".tiff", ".bmp")
TOKEN = re.compile(r"\b(hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|gho_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})")
# built from pieces so that this file does not match itself
MACHINE = [re.compile("/home/" + "mhamdan"), re.compile("/tmp/" + "claude-"), re.compile(r"\b" + "syr" + r"in\b", re.I)]
TEXT_EXT = (".py", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".cff", ".sh", ".ipynb", ".cfg", "")


# the public release does not reference the earlier pre-print; files holding THIRD-PARTY references that
# may legitimately contain such strings (vocabularies, bibliographies of other works) are listed here
EARLIER_REF = re.compile("ar" + "xiv|2412" + r"\.18524|pre" + "print", re.I)
EARLIER_REF_ALLOW = ()      # none in this export (the T5 tokenizer vocabulary lives in the HF staging, not here)


def walk(root):
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in (".git", "__pycache__", ".pytest_cache")]
        for f in files:
            yield os.path.join(d, f)


def scan_text(path, problems, rel):
    try:
        t = open(path, encoding="utf-8").read()
    except (UnicodeDecodeError, OSError):
        return None
    for rx in MACHINE:
        if rx.search(t):
            problems.append(f"{rel}: machine-specific string {rx.pattern!r}")
    m = TOKEN.search(t)
    if m:
        problems.append(f"{rel}: token-like string {m.group(0)[:6]}...")
    return t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--hf-root", default=None)
    ap.add_argument("--data-dir", default=None, help="archives: also check that no benchmark line is embedded")
    ap.add_argument("--max-mb", type=float, default=5.0)
    a = ap.parse_args(argv)
    problems, texts, n, total = [], {}, 0, 0
    listed = ""
    src = os.path.join(a.root, "examples", "SOURCES.md")
    if os.path.exists(src):
        listed = open(src, encoding="utf-8").read()
    rd = lambda *x: open(os.path.join(a.root, *x), encoding="utf-8").read() if os.path.exists(os.path.join(a.root, *x)) else ""
    figures_listed, assets_listed = rd("docs", "figures", "FIGURES.md"), rd("scripts", "build_ood_figure.py")
    for p in walk(a.root):
        rel = os.path.relpath(p, a.root)
        n += 1
        size = os.path.getsize(p)
        total += size
        low = rel.lower()
        if size > a.max_mb * 1e6:
            problems.append(f"{rel}: {size / 1e6:.1f} MB > {a.max_mb} MB")
        if low.endswith(BINARY):
            problems.append(f"{rel}: binary weights / archive / array file")
        if low.endswith(("_preds.tsv", "_t5cache.json")):
            problems.append(f"{rel}: per-line predictions / T5 cache (ground-truth text)")
        if low.endswith(IMAGES + (".pdf", ".svg")):
            base = os.path.basename(rel)
            ok = (rel.startswith("examples" + os.sep) and base in listed) or \
                 (rel.startswith(os.path.join("docs", "figures") + os.sep) and os.path.splitext(base)[0] in figures_listed) or \
                 (rel.startswith(os.path.join("docs", "assets") + os.sep) and base in assets_listed)
            if not ok:
                problems.append(f"{rel}: image not in examples/ (SOURCES.md), docs/figures/ (FIGURES.md) "
                                "or docs/assets/ (scripts/build_ood_figure.py)")
            if base.startswith("qualitative.") or base.startswith("qualitative_caption"):
                problems.append(f"{rel}: paper-only qualitative figure (IAM-HistDB images) must not be released")
        if low.endswith(".ipynb"):
            nb = json.load(open(p, encoding="utf-8"))
            outs = sum(len(c.get("outputs", [])) for c in nb["cells"] if c["cell_type"] == "code")
            if outs:
                problems.append(f"{rel}: {outs} notebook output(s) committed")
        if os.path.splitext(low)[1] in TEXT_EXT:
            t = scan_text(p, problems, rel)
            if t is not None:
                texts[rel] = t
                m = EARLIER_REF.search(t)
                if m and rel not in EARLIER_REF_ALLOW:
                    problems.append(f"{rel}: earlier-work reference {m.group(0)!r} (allowed only in {EARLIER_REF_ALLOW})")
    if a.data_dir:
        import h5py
        gt = set()
        for ds in ("iam", "rimes", "bentham", "saintgall", "washington", "saintgall_nopt"):
            path = os.path.join(a.data_dir, f"{ds}.hdf5")
            if os.path.exists(path):
                with h5py.File(path, "r") as h:
                    for sp in ("train", "valid", "test"):
                        gt.update(g.decode() if isinstance(g, bytes) else g for g in h[sp]["gt"][:])
        long_gt = [g for g in gt if len(g) >= 25]
        for rel, t in texts.items():
            hits = [g for g in long_gt if g in t]
            if hits:
                problems.append(f"{rel}: contains {len(hits)} benchmark transcription line(s), e.g. {hits[0][:30]!r}")
        print(f"checked {len(texts)} text files against {len(long_gt)} benchmark lines (>= 25 chars)")
    if a.hf_root:
        for p in walk(a.hf_root):
            rel = "HF:" + os.path.relpath(p, a.hf_root)
            if os.path.splitext(p)[1].lower() in (".json", ".md", ".txt", ".py", ""):
                scan_text(p, problems, rel)
    print(f"{n} files, {total / 1e6:.2f} MB in {a.root}")
    for x in problems:
        print("FAIL", x)
    print("HYGIENE:", "PASS" if not problems else f"FAIL ({len(problems)})")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
