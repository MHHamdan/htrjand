#!/bin/bash
# 128 x 1632 archives (same recipe as the paper's 68 x 864, larger canvas), combined archive, synthetic lines
set -e
cd "$(dirname "$0")/.."
: "${HTRJAND_DATA_DIR:?set HTRJAND_DATA_DIR to the directory holding the archives}"
export HTRJAND_DATA_DIR
for ds in washington saintgall iam rimes bentham; do
  python -m htrjand.prepare_data --dataset $ds --raw-dir "${HTRJAND_RAW_DIR:-raw}" \
     --size 128 1632 --workers 16 --out $HTRJAND_DATA_DIR/${ds}_128x1632.hdf5 --overwrite
done
python - <<'PY'
import h5py, numpy as np, os
d=os.environ["HTRJAND_DATA_DIR"]
with h5py.File(f"{d}/combined_128x1632.hdf5","w") as out:
    for sp in ["train","valid","test"]:
        dt=[];gt=[]
        for ds in ["iam","rimes","bentham","saintgall","washington"]:
            with h5py.File(f"{d}/{ds}_128x1632.hdf5") as h:
                dt.append(h[sp]["dt"][:]); gt+= [g.decode() if isinstance(g,bytes) else g for g in h[sp]["gt"][:]]
        out.create_dataset(f"{sp}/dt", data=np.concatenate(dt))
        out.create_dataset(f"{sp}/gt", data=[g.encode() for g in gt])
        print(sp, len(gt))
PY
python -m htrjand.synth_archive --fonts synth/gfonts --coverage synth/font_coverage.json \
  --text en=synth/text_en.txt:0.55 fr=synth/text_fr.txt:0.35 la=synth/text_la.txt:0.10 \
  --n 120000 --workers 16 --size 128 1632 --out synth/synth_120k_128x1632.hdf5
echo BUILD DONE
