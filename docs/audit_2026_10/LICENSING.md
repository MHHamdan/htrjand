# Licensing audit: benchmarks, components, release artefacts

> **Release-export note.** Copied from the private research repository (2026-10-17) and adapted to this release. Paths `src/`, `legacy_v2/`, `paper/`, `rev2_package/`, `release/` and `htrjand-jan/` refer to that repository and are not part of this release. Finding F1 (per-line prediction files with ground truth) does not apply: no `*_preds.tsv` is included. Finding F2 applies in reduced form: **a few IAM and Bentham TEST line images appear in four figures of this repository** (`docs/figures/qualitative_examples_readme.png`: IAM and Bentham; `docs/figures/architecture.png`, `attention.png`, `pipeline.png`: IAM). They are shown for non-commercial research illustration under the IAM (CC BY-NC 4.0, Marti & Bunke 2002) and Bentham (CC BY 4.0) terms, with attribution in `docs/figures/FIGURES.md` and in the figure footer; no Saint Gall or Washington image is included.


Audit date: 2026-10-17. Scope: the public release
(GitHub `DocumentRecognitionModels/htrjand`), its model weights, and the
README/demo. This is an engineering reading of published terms, not legal advice.

## 0. Findings and their resolution in this release

| # | Finding | Where | Action |
|---|---|---|---|
| F1 | **895 per-line prediction files carrying the ground-truth `ref` column are tracked in git** (868,912 lines in total, across all five corpora: IAM, RIMES, Bentham, Saint Gall, Washington, plus combined). The test transcriptions of four NC-licensed corpora are redistributed verbatim. | `results/*_preds.tsv`, `legacy_v2/results/postproc/*_preds.tsv` (e.g. `results/bentham_paper_teacher_s1_test_preds.tsv:1` header `idx ref hyp`) | Drop the `ref` column from public prediction files (keep `idx hyp`; scorers can rejoin by index from a local copy of the corpus). The other option is to move them under a `data_derived/` tree carrying a CC BY-NC 4.0 notice. Both are acceptable. Dropping `ref` is cleaner. |
| F2 | Paper figures with real benchmark line images are tracked: `paper/figures/qualitative_examples.{png,pdf}` and `attention_maps.{png,pdf}`, plus the `rev2_package/figures/` copies. They come from IAM test lines (`htrjand-jan/qualitative.py:130`, default `--dataset iam_aachen`). | `paper/figures/`, `rev2_package/figures/` | Allowed (non-commercial research, cited). Add the attribution line from §1.6 to the figure caption and to a `paper/figures/NOTICE`. |
| F3 | The README says RIMES comes from "the ICDAR 2011 line-recognition competition". That distribution channel no longer exists. RIMES is now on Zenodo under CC BY-NC 4.0 (§1.2). | `README.md:247-251` | Point to the Zenodo DOIs (IAM, RIMES, Bentham) and to the FKI page (IAM-HistDB), and state each licence. |
| F4 | Wikipedia text files (`synth/text_{en,fr,la}.txt`, 700k lines) and the font files (`synth/gfonts/`) are on disk but **not tracked**. That is correct, so keep it that way unless you add the notices in §2. | `synth/` | No action. If they are ever shipped, follow §2.3 and §2.4. |
| F5 | The character LM is fitted at run time from training transcriptions plus the Wikipedia text. It is never written to disk as a shipped artefact (`htrjand/ensemble_rescore.py:151-167`). | | No action. If an LM file is published later, it follows the row "n-gram LM" in §1.7. |
| F6 | `postproc/t5data*/pairs_*.tsv` (T5 training pairs, whose targets are GT transcriptions) are untracked. Only `*_meta.json` is tracked. | `postproc/` | Keep the TSVs untracked. |

## 1. Benchmarks

### 1.1 IAM Handwriting Database (FKI, Univ. Bern)

- Official record, now on Zenodo, "migrated from the former FKI website at the University of Bern to make it publicly available with permission of the original authors": <https://zenodo.org/records/19003721>. **Licence field: CC BY-NC 4.0**.
- Terms, verbatim from the record (identical to the historical FKI page <https://fki.tic.heia-fr.ch/databases/iam-handwriting-database>, whose TLS certificate had expired on 2026-10-17 so it could not be fetched directly):
  > "Terms of Use and Citation Request
  > The IAM Handwriting Database is publicly accessible and freely available for non-commercial research purposes. If you are publishing scientific work based on the IAM Handwriting Database, we request you to include a reference to the paper [4]: U. Marti and H. Bunke. The IAM-database: An English Sentence Database for Off-line Handwriting Recognition. Int. Journal on Document Analysis and Recognition, Volume 5, pages 39 - 46, 2002."
- The text content is sentences from the LOB Corpus ("All texts in the IAM database are built using sentences provided by the LOB Corpus [5]"). The transcriptions are therefore third-party text, which is one more reason not to redistribute them in bulk.
- CC BY-NC 4.0 legal code (<https://creativecommons.org/licenses/by-nc/4.0/legalcode.en>), the operative clauses:
  > §2(a)(1) "A. reproduce and Share the Licensed Material, in whole or in part, for NonCommercial purposes only; and B. produce, reproduce, and Share Adapted Material for NonCommercial purposes only."
  > §1(i) "NonCommercial means not primarily intended for or directed towards commercial advantage or monetary compensation."
  > §3(a)(1) requires you to retain "identification of the creator(s)", "a notice that refers to this Public License", "a URI or hyperlink to the Licensed Material to the extent reasonably practicable", to "indicate if You modified the Licensed Material", and to "indicate the Licensed Material is licensed under this Public License".
- Precedent for released weights:
  - `microsoft/trocr-base-handwritten`, "fine-tuned on the IAM dataset", card `license: mit` (<https://huggingface.co/microsoft/trocr-base-handwritten>).
  - `Teklia/pylaia-iam`, `license: mit`. It ships `weights.ckpt` **and `language_model.arpa.gz`**: "An external 6-gram character language model ... is trained on the text from the IAM training set" (<https://huggingface.co/Teklia/pylaia-iam>).
  - `Teklia/IAM-line` (HF dataset), tagged `mit`. **This tag conflicts with the upstream CC BY-NC 4.0, so don't cite it as the licence source.**
  - These are precedents for community tolerance, not a licence grant. The upstream terms stay NC.

### 1.2 RIMES (A2iA / ICDAR 2011 lines)

- Official record: "RIMES, complete", <https://doi.org/10.5281/zenodo.10812725> (2024-03-13), creators Grosicki, Carré, Geoffrois, Augustin, Preteux, Messina. **Licence field: CC BY-NC 4.0**.
- Verbatim from the record:
  > "This dataset was originally collected and prepared in 2007 by the following partners: DGA/CTA/DT/GIP - CEP Arcueil; TSP – ARTEMIS Télécom SudParis; and A2iA SA, as part of the Techno-Vision project. ... After the acquisition of A2iA SA in September 2018, Mitek Systems, Inc. became a legal owner of the dataset, and decided to release it publicly ... under a permissive license in 2024, to encourage open science."
  > "all correspondence is fictitious and there is no PII in the records."
- The Zenodo files are `Images_Courriers.zip`, `images_blocs_de_texte.zip`, `imagettes_mots_cursif.zip` and `imagettes_caracteres.zip`. **The ICDAR-2011 line split used here (`training_2011.xml`, `eval_2011_annotated.xml`; see `htrjand/prepare_data.py` `_rimes`) is not a separate file on that record.** Teklia republishes it as the HF dataset `Teklia/RIMES-2011-line`, tagged `mit`. `Teklia/pylaia-rimes` (MIT) ships weights plus `language_model.arpa.gz`.
- Recommendation: treat the line split as CC BY-NC 4.0, the rights holder's licence, and cite Zenodo 10812725 as the source.

### 1.3 Bentham (ICFHR 2014 HTRtS, tranScriptorium)

- Official record: "Bentham Dataset R0", <https://doi.org/10.5281/zenodo.44519> (2016-01-08, creator J. A. Sánchez). **Licence field: CC BY 4.0**. Description, verbatim:
  > "The Bentham collection consists of a set of images of a collection of works on law and moral philosophy written by the philosopher Jeremy Bentham. This dataset is free available for research purposes and it is provided into two parts: the images and the GT."
  The README on the record asks for citation of Gatos et al., DAS 2014 ("Ground-truth production in the tranScriptorium project") and Sánchez et al., ICFHR 2014 (HTRtS).
- The companion ICDAR 2015 HTRtS record (<https://zenodo.org/records/248733>) is also CC BY 4.0.
- Upstream rights: the page images are UCL Special Collections' Bentham Papers. Transcripts on the Transcribe Bentham Transcription Desk are stated to be released under **CC BY-NC-ND 4.0** (search-index text of <https://transcribe-bentham.ucl.ac.uk/td/Transcribe_Bentham>; the page returned HTTP 403 to direct fetch). The dataset publisher applied CC BY 4.0 to the packaged GT. That is the operative grant for this dataset, but the chain of title is not perfectly clean.
- CC BY 4.0 allows redistribution, including commercial use, with attribution. It is the most permissive of the five.

### 1.4 Washington (IAM-HistDB)

- Source: IAM Historical Document Database, FKI, <https://fki.tic.heia-fr.ch/databases/iam-historical-document-database> and `/washington-database`. The pages could not be fetched directly on 2026-10-17 (expired TLS certificate, and the Internet Archive was offline). The terms below are quoted from the search-engine index of that page. Re-check them before relying on the wording:
  > "Before you download one of the IAM-HistDB data sets, you need to register ... The IAM-HistDB data sets may be used for non-commercial research and teaching purposes only."
- No CC licence or Zenodo migration was found (Zenodo search, 2026-10-17). Under these bespoke terms, **redistribution is not granted**.
- Underlying pages: the George Washington Papers at the Library of Congress are 18th-century public domain. The *binarised, normalised line images and the GT* are FKI's work product under the IAM-HistDB terms.
- Citation normally given: A. Fischer, A. Keller, V. Frinken, H. Bunke, "Lexicon-free handwritten word spotting using character HMMs", Pattern Recognition Letters 33(7), 2012. A. Fischer, "IAM-HistDB – A Dataset of Handwritten Historical Documents", in *Handwritten Historical Document Analysis, Recognition, and Retrieval*, World Scientific, 2020 (doi:10.1142/9789811203244_0002).

### 1.5 Saint Gall (IAM-HistDB; Cod. Sang. 562)

- Dataset terms are the same IAM-HistDB terms as §1.4 (non-commercial research and teaching only, registration, no redistribution grant).
- Source images: St. Gallen, Stiftsbibliothek, Cod. Sang. 562 (Walahfrid Strabo, *Vitae sancti Galli et Otmari*, 890-900), e-codices. The IIIF manifest <https://www.e-codices.ch/metadata/iiif/csg-0562/manifest.json> declares `"license": "https://creativecommons.org/licenses/by-nc/4.0/"`.
- e-codices terms (<https://www.e-codices.unifr.ch/en/about/terms>):
  > "use of our digitally reproduced images for non-commercial purposes is free"; commercial use requires "prior written permission of the manuscript owner library".
  The required credit form is "Location, + library, + shelfmark, + page number (www.e-codices.ch)". For this manuscript that is "St. Gallen, Stiftsbibliothek, Cod. Sang. 562, p. N (www.e-codices.ch)".
- Citation: A. Fischer, V. Frinken, A. Fornés, H. Bunke, "Transcription alignment of Latin manuscripts using hidden Markov models", HIP 2011, and the IAM-HistDB chapter above.

### 1.6 Attribution lines to paste

- IAM: "IAM Handwriting Database (U. Marti, H. Bunke, IJDAR 5:39-46, 2002), doi:10.1007/s100320200071, CC BY-NC 4.0, https://zenodo.org/records/19003721."
- RIMES: "RIMES database (Grosicki et al.; A2iA / Mitek Systems), doi:10.5281/zenodo.10812725, CC BY-NC 4.0."
- Bentham: "Bentham Dataset R0 (tranScriptorium; Sánchez et al., ICFHR 2014), doi:10.5281/zenodo.44519, CC BY 4.0. Images: UCL Special Collections, Bentham Papers."
- Washington: "IAM-HistDB Washington Database (FKI, Univ. Bern/Fribourg), used under the IAM-HistDB non-commercial research terms. Source pages: George Washington Papers, Library of Congress."
- Saint Gall: "IAM-HistDB Saint Gall Database (FKI). Source: St. Gallen, Stiftsbibliothek, Cod. Sang. 562 (www.e-codices.ch), CC BY-NC 4.0."

### 1.7 Recommendation table

"OK" means permitted under the stated terms with the attribution in §1.6. "NC" means permitted for non-commercial research only, with the CC BY-NC notice. "No" means not granted, or not advisable.

| Use | IAM | RIMES | Bentham | Washington | Saint Gall |
|---|---|---|---|---|---|
| (a) Line images committed to a public GitHub repo | **No** in bulk. CC BY-NC permits NC sharing, but the copy would sit inside an MIT tree and confuse downstream users. A handful under a separate NC notice is permissible. | **No** in bulk (same reasoning). A handful under an NC notice is OK. | OK (CC BY 4.0). Still prefer linking to Zenodo. | **No.** The IAM-HistDB terms grant no redistribution. | **No** (IAM-HistDB terms; the e-codices source is also NC). |
| (b) A few example lines in a paper figure | OK + cite Marti & Bunke 2002. See the note on commercial journals below. | OK + cite | OK + cite | OK (research use, standard practice) + cite Fischer et al. | OK + cite + e-codices credit line |
| (c) A few lines in README / demo | NC, with attribution. Prefer the OOD set (§3). | NC, with attribution. Prefer OOD. | OK | **Avoid.** Use the PD Library of Congress page instead, or the OOD set. | **Avoid** the HistDB line images. An e-codices crop with credit is NC-OK. |
| (d) Release trained weights | NC. Precedent: TrOCR (MIT) and pylaia-iam (MIT). Licence our weights **CC BY-NC 4.0**. | NC. Precedent: pylaia-rimes (MIT) | OK | NC ("research and teaching") | NC |
| (e1) Character n-gram LM file fitted on training transcriptions | NC. An order-10 LM on about 6k lines is close to a copy of the text. Precedent: pylaia ships 6-gram ARPA files. Ship only under CC BY-NC, or not at all. Currently it is not shipped (F5). | NC | OK (CC BY) | **No** file. Fit at run time. | **No** file. Fit at run time. |
| (e2) Per-line prediction files containing GT text | **Strip `ref`** (F1). | Strip `ref` | OK to keep | **Strip `ref`** | **Strip `ref`** |
| Transcription text in bulk (any form) | No (LOB-derived, NC) | No (NC) | OK (CC BY) | No | No |

Commercial journal note for (b): Pattern Recognition (Elsevier) is a commercial publisher. Including a few cited dataset samples in a research article is standard scholarly practice for all five corpora. If the article goes open access under CC BY, mark the figure as third-party content: "Example images from [dataset], licensed CC BY-NC 4.0; not covered by the article licence."

Combined-corpus weights: they are trained on all five corpora, so they inherit the most restrictive terms, NC research only. The only weights that can be released permissively are Bentham-only or synthetic-only weights.

## 2. Components

| Component | Licence (verified) | Obligation for this release |
|---|---|---|
| `google-t5/t5-small` weights + config | Apache-2.0 (HF card `license: apache-2.0`; pretraining data C4) | A fine-tuned T5 may be redistributed. Include the Apache-2.0 text. Apache §4(b) requires modified files to carry "prominent notices stating that You changed the files", so state "fine-tuned from google-t5/t5-small" in the model card and NOTICE. |
| SentencePiece model `spiece.model` (inside t5-small) | Distributed in the same Apache-2.0 HF repo. The SentencePiece library is Apache-2.0 (github google/sentencepiece). | Ship it unmodified inside the fine-tuned checkpoint. It is covered by the same Apache-2.0 notice. |
| Wikipedia text (en/fr/la), used in `synth/text_*.txt`, the char LM, synthetic lines and T5 corruption pairs | CC BY-SA 4.0 (Wikipedia: "If you make modifications or additions to the page you re-use, you must license them under the Creative Commons Attribution-Share-Alike License 4.0 or later"; attribution by "a hyperlink ... or URL to the page or pages you are re-using ... or a list of all authors"). CC BY-SA 4.0 §3(b)(1): "The Adapter's License You apply must be a Creative Commons license with the same License Elements, this version or later, or a BY-SA Compatible License." | See (i) to (iii) below. |
| Google Fonts handwriting faces: 78 OFL-1.1 (`synth/gfonts/ofl/`), 10 Apache-2.0 (`synth/gfonts/apache/`) | OFL-1.1 / Apache-2.0 | **Rendered images are unrestricted.** OFL FAQ 1.1: "Authors of fonts released under the OFL allow you to use their font software as such for any kind of design work". FAQ 1.2: "You remain the author and copyright holder of that newly derived graphic or object" (<https://openfontlicense.org/ofl-faq/>). Apache-2.0 places no condition on output either. If the font *files* are shipped, each must keep its `OFL.txt` / `LICENSE.txt`, and OFL fonts may not be sold by themselves. Recommend not shipping them: list family names plus a fetch script from github.com/google/fonts. |
| PyTorch 2.9.1 | BSD-3-Clause (GitHub reports NOASSERTION because of a custom header, but the text is BSD-3) | Notice only if you vendor it. pip dependency, no action. |
| Gradio | Apache-2.0 | No action for a pip dependency. A demo Space may be licensed as you choose. |
| huggingface_hub, transformers | Apache-2.0 | No action. HF Hub cards take a `license:` field: use `cc-by-nc-4.0` for the weights. Optional `extra_gated_prompt` to make users acknowledge the NC dataset terms. |
| NumPy, h5py | BSD-3-Clause | none |
| OpenCV (`opencv-python-headless`) | Apache-2.0 (OpenCV ≥4.5), MIT packaging | none |
| editdistance | MIT | none |
| Legacy code under `src/` (Arthur Flôr) | MIT, notice retained in `LICENSE` | Keep the retained notice. |

Wikipedia obligations by artefact:

- **(i) An n-gram LM file built from Wikipedia text.** A character 9/10-gram model stores verbatim substrings and their counts. Treat a shipped LM file as Adapted Material, licence it **CC BY-SA 4.0**, and attribute it as "Contains text from Wikipedia (en/fr/la), dump YYYY-MM-DD, https://dumps.wikimedia.org, CC BY-SA 4.0." If it also mixes in NC training transcriptions, BY-SA and NC cannot be combined cleanly: SA requires the same licence elements, and BY-NC-SA is not BY-SA. **So never publish an LM that mixes Wikipedia and IAM/RIMES/HistDB text.** Publish a Wikipedia-only LM (CC BY-SA), or none. Today none is published (F5).
- **(ii) T5 fine-tuned partly on corrupted Wikipedia lines.** Creative Commons' guidance (<https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/>): BY and SA "are triggered only when works or adaptations of works are publicly shared"; "If AI models or outputs are based on ShareAlike content and they will be shared publicly, following the ShareAlike condition would require AI developers to use the same CC license". Whether the weights are an adaptation is unsettled. Common practice is to release models trained on Wikipedia under permissive licences (BERT, T5, Apache-2.0). The T5 corrector is *also* trained on GT transcriptions (`postproc/t5data/pairs_train.tsv` targets), so the binding constraint is the dataset NC terms, not BY-SA. Recommendation: CC BY-NC 4.0 for the corrector, and a model-card line "fine-tuned on synthetic corruptions of Wikipedia text (CC BY-SA 4.0) and of benchmark training transcriptions".
- **(iii) A text sample file** (`synth/text_*.txt`, verbatim sentences). Must be distributed under CC BY-SA 4.0, with attribution (dump date + URL) and the licence text, in its own directory with a `LICENSE` file, outside the MIT grant. Better: ship the extraction script and dump date instead of the 700k lines.

### Licence recommendation

- **Code: keep MIT** (`LICENSE`, retains the Flôr notice). It is compatible with every dependency above. Add one paragraph: "The MIT licence covers source code only. Datasets, transcriptions derived from them, model weights and third-party text are under the licences listed in docs/audit_2026_10/LICENSING.md."
- **Weights (all released checkpoints and the T5 corrector): CC BY-NC 4.0.** Add a model card that lists the five datasets with their licences (§1.6), Wikipedia (CC BY-SA 4.0) and t5-small (Apache-2.0).

## 3. Out-of-distribution demo images

See `examples/SOURCES.md (copied from the release assets; originals not included)` for the per-image URL, licence, attribution and transcription. All are public domain or CC0. None comes from the five benchmarks, nor from the Library of Congress George Washington Papers or e-codices, which are the benchmark sources.

## 4. Open items

1. FKI site (IAM-HistDB) could not be fetched (expired certificate, Internet Archive offline). Re-read the Washington and Saint Gall terms when it is back. Or email FKI/HES-SO (Andreas Fischer) to ask (a) whether a few line images may appear in a README and (b) whether NC weight release is acceptable.
2. RIMES ICDAR-2011 line XMLs: record where the local copy came from (A2iA registration, 2011, or Teklia). Rights rest with Mitek under CC BY-NC 4.0 either way.
3. Bentham: the Transcribe Bentham CC BY-NC-ND statement could not be fetched directly (HTTP 403). The Zenodo CC BY 4.0 grant is what we rely on.
