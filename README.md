# HMM activity recognition

A simple, from-scratch Hidden Markov Model (HMM) in numpy (implements the forward, backward, Viterbi, Baum-Welch algorithm). It is based on four activities I recorded using the application Sensor Logger: Still, Standing, Walking and Jumping, which I decoded from the data from the accelerometer and gyroscope of my phone. No HMM library is used. You must have numpy, pandas, scipy, matplotlib and seaborn.
Author: Your Name | Course: Your Course Name

The results on unseen data (Viterbi)

Test set | Windows scored | Overall accuracy |
|---|---|---|
| 2 required unseen files (`jumping_test_01`, `still_test_01`, iPhone 11 Pro) | 133 | 0.962 |
| 2 unseen files + multi-activity clip | 163 | 0.920 |
| multi-activity clip alone | 30 | 0.733 |

Walking and Standing are only scored on the multi-activity clip (6 and 8 windows) because there is no file of these activities.The Walking and Standing files do not exist, so they are only scored on the multi-activity clip. This clip shows a lot of jumps which are not as sharp as in training, and are primarily interpreted as Walking. The report spells out the details and limits.

## How to run

```
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace hmm_activity_recognition.ipynb
cd ../report
python build_report.py
```

It reads the two files `data/raw/`, `data/recordings_manifest.csv`, and writes the following files: `data/processed/`, `data/chunks/`, the feature files, and `figures/` and `figures/results.json. Random seed is 42. These outputs are used to rebuild the report pdf in the report script.

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

Recordings: Walking, jumping (2 clips), standing at waist level, still (gravity channel on), 2 unseen test clips (still, jumping) from different phone, 1 clip of all 4 activities.
* All signals are resampled to 50 Hz, trimmed by 3 s at each end and split into 10 s chunks. Windows are 2 s with 50% overlap and remain within a chunk.
`jumping_test_01` was taken at the same time as `jumping_train_01`, so I cut the training clip down to exclude the common area.
* Segments of the multi-activity clip: Still 0-10 s, pick-up 10-12 s, Standing 12-21 s, Jumping 21-31.5 s, Walking from 31.5 s.
There is only one activity in each training clip, and the training sequences are random concatenations of 10 s clips. The off-diagonal transition probabilities do not reflect actual behaviour, but rather are a consequence of that choice.

## Videos

Standing:https://drive.google.com/file/d/14bXi_YKq8MsvZhjWOkLVvNvk-Px5cNnc/view?usp=sharing
Walking:https://drive.google.com/file/d/1DNB2eZrVowvWcW2GhMEnhiuI1NYyTArG/view?usp=sharing
Jumping:https://drive.google.com/file/d/1MJbGbg7KquVw6Vf7bx5yNw0bP62Fzic2/view?usp=sharing
Still:https://drive.google.com/file/d/1AxzBbmKk0YBzZTilwP2R7I1w8gaNq4o2/view?usp=sharing 
