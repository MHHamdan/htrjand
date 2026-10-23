#!/usr/bin/env python3
"""Download the inputs of the synthetic lines and of the character LMs into synth/.

    python scripts/fetch_synth_resources.py                    # text + all 106 fonts -> synth/
    python scripts/fetch_synth_resources.py --fonts-only --limit 3
    python scripts/fetch_synth_resources.py --text-source /path/to/local/htrjand-charlm

Text: synth_text/text_{en,fr,la}.txt.gz of the character-LM repository (default MHamdan/htrjand-charlm,
CC BY-SA): English = WikiText-103 raw (Salesforce/wikitext), French / Latin = Wikipedia article text via the
wikimedia/wikipedia dataset (snapshot not recorded). Written as synth/text_<lang>.txt and checked against the
SHA-256 in configs/synth_text_manifest.json, so the files are the exact inputs of the experiments.

Fonts: the Google Fonts handwriting faces listed in configs/synth_fonts.json, downloaded by path from the pinned
commit of https://github.com/google/fonts into synth/gfonts/<licence>/<family>/<file> and checked against their
SHA-256 (OFL-1.1 / Apache-2.0; the font files are not redistributed here). synth/font_coverage.json (the
`--coverage` input of htrjand.synth_archive) is written from the same manifest, with paths relative to the
repository root, so `python -m htrjand.synth_archive --fonts synth/gfonts --coverage synth/font_coverage.json ...`
reproduces the archive build of scripts/build_c128.sh.
"""
import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download(url, dst, tries=3):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(dst + ".part", "wb") as f:
                shutil.copyfileobj(r, f)
            os.replace(dst + ".part", dst)
            return dst
        except OSError as e:
            if k == tries - 1:
                raise RuntimeError(f"download failed: {url}: {e}") from e
            time.sleep(2 * (k + 1))


def fetch_fonts(out, limit=None):
    man = json.load(open(os.path.join(ROOT, "configs", "synth_fonts.json"), encoding="utf-8"))
    fonts = man["fonts"][:limit] if limit else man["fonts"]
    coverage = {}
    for i, f in enumerate(fonts, 1):
        dst = os.path.join(out, "gfonts", f["path"])
        if not (os.path.exists(dst) and sha256(dst) == f["sha256"]):
            url = f"https://raw.githubusercontent.com/google/fonts/{man['commit']}/" + urllib.parse.quote(f["path"])
            download(url, dst)
            got = sha256(dst)
            if got != f["sha256"]:
                os.remove(dst)
                raise RuntimeError(f"{f['path']}: SHA-256 {got[:12]} != manifest {f['sha256'][:12]}")
        coverage[os.path.relpath(dst, ROOT) if dst.startswith(ROOT) else dst] = f["coverage"]
        print(f"  font {i}/{len(fonts)} {f['path']} ({f['licence']}) ok", flush=True)
    path = os.path.join(out, "font_coverage.json")
    json.dump(coverage, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    return path, len(fonts)


def fetch_text(out, source, revision=None, langs=("en", "fr", "la")):
    man = json.load(open(os.path.join(ROOT, "configs", "synth_text_manifest.json"), encoding="utf-8"))
    for lang in langs:
        e = man[lang]
        dst = os.path.join(out, f"text_{lang}.txt")
        if os.path.exists(dst) and sha256(dst) == e["sha256_text"]:
            print(f"  text {lang}: present", flush=True)
            continue
        if os.path.isdir(source):
            gz = os.path.join(source, e["file"])
        else:
            from huggingface_hub import hf_hub_download
            gz = hf_hub_download(repo_id=source, filename=e["file"], revision=revision)
        os.makedirs(out, exist_ok=True)
        with gzip.open(gz, "rb") as fi, open(dst + ".part", "wb") as fo:
            shutil.copyfileobj(fi, fo)
        got = sha256(dst + ".part")
        if got != e["sha256_text"]:
            os.remove(dst + ".part")
            raise RuntimeError(f"text_{lang}: SHA-256 {got[:12]} != manifest {e['sha256_text'][:12]}")
        os.replace(dst + ".part", dst)
        print(f"  text {lang}: {e['lines']:,} lines ({e['source']})", flush=True)


def main(argv=None):
    from htrjand import release as REL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "synth"))
    ap.add_argument("--text-source", default=REL.DEFAULT_CHARLM, help="Hub repo id or local directory of the LM repo")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--fonts-only", action="store_true")
    ap.add_argument("--text-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="first N fonts only (tests)")
    a = ap.parse_args(argv)
    try:
        if not a.fonts_only:
            fetch_text(a.out, a.text_source, a.revision)
        if not a.text_only:
            path, n = fetch_fonts(a.out, a.limit)
            print(f"{n} fonts -> {os.path.join(a.out, 'gfonts')}; coverage -> {path}")
    except (RuntimeError, OSError) as e:
        sys.exit(f"fetch_synth_resources: {e}")


if __name__ == "__main__":
    main()
