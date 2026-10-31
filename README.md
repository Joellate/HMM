# HMM activity recognition

A Hidden Markov Model written from scratch in numpy (forward, backward, Viterbi, Baum-Welch). It decodes four activities, **Still, Standing, Walking and Jumping**, from phone accelerometer and gyroscope data that I recorded with Sensor Logger. No HMM library is used. The code needs numpy, pandas, scipy, matplotlib and seaborn.

Author: [your name] | Course: [course name]

## Results on unseen data (Viterbi)

| Test set | Windows scored | Overall accuracy |
|---|---|---|
| 2 required unseen files (`jumping_test_01`, `still_test_01`, iPhone 11 Pro) | 133 | 0.962 |
| 2 unseen files + multi-activity clip | 163 | 0.920 |
| multi-activity clip alone | 30 | 0.733 |

The two unseen files only contain Still and Jumping, so Walking and Standing are scored on the multi-activity clip (6 and 8 windows). The jumps in that clip are much gentler than in training and are mostly decoded as Walking. The report has the details and limits.

## How to run

```
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace hmm_activity_recognition.ipynb
cd ../report
python build_report.py
```

The notebook reads `data/raw/` and `data/recordings_manifest.csv`, then writes `data/processed/`, `data/chunks/`, the feature files, `figures/` and `figures/results.json`. Random seed is 42. The report script rebuilds `report/report.pdf` from those outputs.

## Folder layout

```
data/
  raw/                 labelled Sensor Logger CSVs: <activity>_<train|test>_<nn>_<sensor>.csv
  processed/           50 Hz, trimmed recordings (made by the notebook)
  chunks/              every 10 s chunk as its own labelled CSV (made by the notebook)
  recordings_manifest.csv   original file name, label, role and device of each recording
  qc_table.csv, chunk_table.csv, features_*.csv, hmm_params.npz
notebooks/hmm_activity_recognition.ipynb
src/data_utils.py      loading, resampling, trimming, chunking
src/features.py        windows, features, Z-score
src/hmm.py             GaussianHMM (forward, backward, Viterbi, Baum-Welch) and sanity tests
figures/               figures, metric tables, results.json
report/                report.pdf and build_report.py
```

## Data notes

* Recordings: walking, jumping (two clips), standing at waist level, still on a flat surface (gravity channel on), two unseen test clips (jumping and still) from a different phone, and one clip with all four activities.
* All signals are resampled to 50 Hz, trimmed by 3 s at each end and split into 10 s chunks. Windows are 2 s with 50% overlap and stay inside a chunk.
* `jumping_test_01` was recorded at the same time as `jumping_train_01`, so I removed the overlapping part of the training clip.
* Segments of the multi-activity clip: Still 0-10 s, pick-up 10-12 s, Standing 12-21 s, Jumping 21-31.5 s, Walking from 31.5 s.
* No training clip contains more than one activity, so the training sequences are random concatenations of 10 s chunks. The off-diagonal transition probabilities come from that choice and not from real behaviour.

## Videos

[Standing: link] [Walking: link] [Jumping: link] [Still: link]
