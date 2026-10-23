# Data splits: official partitions and overlap check

All models of this repository are trained, selected and tested on the **official writer-independent partitions**
built by `htrjand/prepare_data.py` (IAM: FKI large writer-independent task, train / validationset1+2 / test;
RIMES: ICDAR 2011 lines, whose distribution has no validation partition, so validation = the last 10 % of the official training lines in file order (1,134 lines; training keeps the other 10,192); Bentham: ICFHR 2014 R0; Saint Gall and Washington:
IAM-HistDB sets, Washington cv1). Split sizes (train / valid / test): IAM 6,161 / 1,840 / 1,861; RIMES 10,192 / 1,134 /
778; Bentham 9,194 / 1,415 / 860; Saint Gall 468 / 235 / 707; Washington 325 / 168 / 163 (`tests/test_data_archives.py`).

The overlap check below is the official-split part of `scripts/overlap_audit.py` (output `results/overlap_audit.json`,
example strings redacted; run: `python scripts/overlap_audit.py --data-dir $HTRJAND_DATA_DIR --raw-dir $HTRJAND_RAW_DIR`).
Archives checked per corpus: `official68` (`<c>.hdf5`, 68×864), `official128` (`<c>_128x1632.hdf5`, the archives of the
released models; Saint Gall: `saintgall_nopt_128x1632.hdf5`) and, for Saint Gall, `official68_nopt`. The script also
compares other archive versions when they are present in the data directory; those comparisons are not part of this
release.

## 1. Method

**Categories measured** (`scripts/overlap_audit.py:1-37`, `compare()` at `:190`):

| | Test | Implementation |
|---|---|---|
| A | identical sample ID | The archives store no IDs. `raw_ids()` (`:135`) replays the official partition readers of `htrjand/prepare_data.py` on `raw/`, keeping the builder's order and dropping empty labels. `render_hashes()` (`:155`) re-renders every raw line with the builder's `preprocess()` at 68×864 and at 128×1632. Each archive row is then mapped to the raw line with the same pixel MD5. |
| B | identical image array | `md5(str(shape) + uint8 bytes)` (`:94`). Same canvas only. |
| C | near-duplicate image | `feature()` (`:71`) does four things: crops to the rendered line (rows and columns that are not entirely the pad value), subtracts a local box-mean background and keeps the ink residual, resizes to 32×256 with INTER_AREA, then mean-centres and L2-normalises. Two rows are near-duplicates if their cosine is ≥ τ = 0.88 and their log aspect ratios differ by ≤ 0.15. The test is exhaustive (every query against every reference) and works across canvases. |
| D | identical normalised text | NFC + whitespace collapse, case-sensitive; also casefold. |
| E | text-only repeat | A D match where no reference line with that text is a B or C match of the query image. |

**Validation of A.**
- Every row of every 68×864 and 128×1632 official archive maps to exactly one raw line ID: 0 ambiguous, 0 unmapped, 0 render errors. No two raw lines share an MD5.
- Each mapped row's label equals the raw label, with one expected exception. In the Saint Gall nopt archives the training labels were deliberately changed (`scripts/build_saintgall_nopt.py`), so only 29/468 train labels match there.
- Archive row order equals builder order for all official archives except `saintgall.hdf5`. That archive was written in glob order; mapping through MD5 makes this irrelevant.
- Category A is therefore exact, not inferred.

**Validation of C** (`threshold_validation` in the JSON).
- Positives are the same raw line in `official68` and in `official128`. That is two canvases, never byte-identical.
- Negatives are official test against official train in `official68`. These are disjoint by ID.

| corpus | positives | positives < τ (recall) | negatives max cosine | negatives ≥ τ |
|---|---|---|---|---|
| IAM | 9,862 | 14 (99.86 %) | 0.351 | 0 |
| RIMES | 12,104 | 100 (99.17 %) | 0.425 | 0 |
| Bentham | 11,469 | 3 (99.97 %) | 0.837 (single-letter lines "C" vs "C") | 0 |
| Saint Gall | 1,410 | 0 (100 %) | 0.413 | 0 |
| Washington | 656 | 3 (99.54 %) | 0.337 | 0 |

- Same-canvas byte-identical pairs give cosine 1.0, so C recall equals B recall within one canvas.
- Across canvases, C misses up to 0.8 % of true pairs (RIMES). C is therefore a slight lower bound when compared across canvases; A is the exact count.
- A first version of the feature without background removal produced 92 false positives on Bentham. It was replaced before the reported run.

## 2. Result on the official splits

There is no image-level overlap between any two partitions of any official archive (A = B = C = 0) for `official68`, `official128` or `saintgall_nopt*`, on all five corpora. There are also no within-split byte duplicates. **The gate passes.**

The remaining overlaps are text-only repeats: the same string on different images, category E. All D matches are E matches.

| corpus | train∩valid (D cs / ci) | train∩test (D cs / ci) | valid∩test (D cs / ci) | Typical strings |
|---|---|---|---|---|
| IAM | 0 / 0 | 0 / 0 | 0 / 0 | – |
| RIMES | 150 / 160 | 120 / 122 | 59 / 62 | "Cordialement,", "salutations distinguées.", "distinguées." |
| Bentham | 131 / 136 | 66 / 69 | 42 / 44 | "C", "2", "in" (marginal letters, numbers) |
| Saint Gall | 0 / 0 | 0 / 0 | 0 / 0 | – |
| Washington | 2 / 2 | 4 / 5 | 4 / 4 | "GW", "Regiment.", <one 48-character label, not reproduced (IAM-HistDB terms)> |

The counts are queries of the second split whose text occurs in the first. They are identical for `official68` and `official128`. These repeats are a property of the benchmarks and are not leakage.

