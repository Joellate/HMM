"""Builds report/report.pdf from figures/ and figures/results.json (every number comes from the executed notebook)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether

REPO = Path(__file__).resolve().parents[1]; FIG = REPO / "figures"
R = json.load(open(FIG / "results.json")); man = pd.read_csv(REPO / "data/recordings_manifest.csv")
ACT = ["Still", "Standing", "Walking", "Jumping"]

body = ParagraphStyle("b", fontName="Helvetica", fontSize=9, leading=11.4, alignment=TA_JUSTIFY, spaceAfter=4)
h1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=12, leading=14, spaceBefore=7, spaceAfter=4, keepWithNext=1, textColor=colors.HexColor("#1f3a5f"))
cap = ParagraphStyle("c", fontName="Helvetica-Oblique", fontSize=7.8, leading=9.4, spaceAfter=5, alignment=TA_JUSTIFY)
tcap = ParagraphStyle("tc", parent=cap, spaceAfter=2, spaceBefore=3)
cell = ParagraphStyle("cell", fontName="Helvetica", fontSize=7.8, leading=9)
title = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=15, leading=18, textColor=colors.HexColor("#1f3a5f"))
sub = ParagraphStyle("s", fontName="Helvetica", fontSize=9, leading=11, textColor=colors.HexColor("#444444"), spaceAfter=3)
PH = lambda s: f'<font backColor="#fff2a8">[{s}]</font>'
P = lambda t, st=body: Paragraph(t, st)
fnum = lambda x: "undefined" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"

def img(name, w_cm):
    w, h = PILImage.open(FIG / name).size
    return Image(str(FIG / name), width=w_cm * cm, height=w_cm * cm * h / w)

def tbl(rows, widths, header=True, fs=7.8, align="CENTER"):
    t = Table([[Paragraph(str(c), cell) if isinstance(c, str) and len(str(c)) > 22 else c for c in r] for r in rows], colWidths=[w * cm for w in widths])
    st = [("FONTSIZE", (0, 0), (-1, -1), fs), ("GRID", (0, 0), (-1, -1), .4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("ALIGN", (1, 0), (-1, -1), align), ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]
    if header: st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe8f3")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(st)); return t

def metrics_rows(tab):
    rows = [["Activity", "Samples (windows)", "Sensitivity", "Specificity", "Accuracy (one-vs-rest)"]]
    for r in tab: rows.append([r["Activity"], str(r["Samples"]), fnum(r["Sensitivity"]), fnum(r["Specificity"]), fnum(r["Accuracy (one-vs-rest)"])])
    return rows

acc = R["acc"]; A = np.array(R["A"]); pf = R["per_file"]["2 unseen + multi-activity"]
a_u = acc["Viterbi + Baum-Welch (main) | 2 unseen files"]; a_h = acc["Viterbi + Baum-Welch (main) | 2 unseen + multi-activity"]
a_m = acc["Viterbi + Baum-Welch (main) | multi-activity only"]
fw_h = acc["Frame-wise Gaussian (no transitions) | 2 unseen + multi-activity"]; fw_m = acc["Frame-wise Gaussian (no transitions) | multi-activity only"]
wc = R["window_counts"]; n_u = sum(wc["test_unseen"].values()); n_h = n_u + sum(v for k, v in wc["test_multi"].items() if k != "transition")
jv = R["jump_median_var"]; jg = R["jump_median_gyr"]; mf = R["median_feats"]
dwell = 1 / (1 - np.diag(A).mean())

S = []
S += [P("Modeling Human Activity States Using Hidden Markov Models", title),
      P(f"Formative 2 (individual report) &nbsp;|&nbsp; Author: {PH('your name')} &nbsp;|&nbsp; Student ID: {PH('your ID')} &nbsp;|&nbsp; Repository: {PH('GitHub link')}", sub)]

# ---------------------------------------------------------------- 1
S += [P("Background and Motivation", h1),
      P("Smartphones carry inertial sensors that make it possible to recognise what a person is doing without extra hardware, which supports fitness and step tracking, "
        "monitoring of sedentary behaviour, and context-aware services such as inactivity or fall-risk alerts. Raw accelerometer and gyroscope streams are noisy and sequential: "
        "an activity lasts seconds to minutes, and people rarely switch between standing, walking and jumping every second. A Hidden Markov Model (HMM) fits this structure naturally, "
        "because it separates the hidden activity, which we want to know, from the noisy sensor features, which we observe, and it encodes how likely one activity is to follow another. "
        "This report builds a four-state HMM (Still, Standing, Walking, Jumping) from scratch with numpy, trains it on windowed sensor features, and evaluates it on recordings that were never used for training.")]

# ---------------------------------------------------------------- 2
S += [P("Data Collection and Pre-processing", h1),
      P("Accelerometer and gyroscope data were recorded with the Sensor Logger app for this individual assignment. The training clips and the continuous multi-activity clip come from an iPhone 11; "
        "the two required unseen test clips come from a different phone, an iPhone 11 Pro. Table 1 lists every recording with its native sampling rate (estimated from the timestamps).")]
rows = [["Recording", "Label", "Role", "Device", "Native rate (Hz)", "Duration (s)"]]
for _, r in man.iterrows():
    lab = r.label if "Still>" not in r.label else "Still, Standing, Jumping, Walking"
    role = {"train": "train", "test": "unseen test", "test (all 4 activities)": "held-out test"}.get(r.role, r.role)
    rows.append([r.recording, lab, role, r.device, f"{r.native_hz:.1f}", f"{r.duration_s:.1f}"])
S += [P("<b>Table 1.</b> Recordings, devices and original sampling rates (individual data, so the table lists devices instead of group members).", tcap),
      tbl(rows, [3.5, 4.6, 2.3, 2.3, 2.5, 2.1], fs=7.5)]
S += [Spacer(1, 4),
      P(f"<b>Harmonising the sampling rate.</b> Both phones record at about 100 Hz with slightly irregular timestamps (100.1 Hz and 99.8 Hz). Every signal is linearly interpolated onto an exact 100 Hz grid, "
        "passed through a zero-phase FIR anti-alias filter and decimated by two to a common <b>50 Hz</b>. 50 Hz keeps content up to 25 Hz, well above walking and jumping cycles (about 1 to 4 Hz), "
        "and the anti-alias filter prevents folding of higher-frequency vibration. Three seconds are trimmed from both ends of each clip to remove phone-handling transients, "
        "and each clip is split into non-overlapping <b>10 s chunks</b> (remainders under 5 s dropped). Windows are cut inside chunks only, and all chunks of one recording stay on the same side of the train/test split. "
        "The data had no missing values, no timestamp gaps above 0.01 s, and the same number of accelerometer and gyroscope samples. Sensor Logger writes axes in the order z, y, x, which the loader corrects. "
        "The accelerometer is gravity-free (m/s<super>2</super>); the clip recorded with gravity enabled keeps gravity in a separate channel that is not used."),
      P(f"<b>Leakage check.</b> The timestamps show that the unseen jumping clip was recorded at the same moment as the training clip jumping_train_01 (by the second phone; overlap {R['leak_overlap_s']} s). "
        f"To keep the test data unseen, the first {R['leak_cut_s']} s of that training clip (the overlap plus a 1 s margin) were removed from training. "
        "The test files differ from the training files in the phone used, but all recordings were made on one day within one project, so participant, placement and environment diversity is minimal; this limits how far results can be generalised."),
      P("<b>Windowing.</b> Windows are 2 s long (100 samples at 50 Hz) with 50% overlap. Walking steps and jumps repeat at roughly 1.5 to 4 Hz, so a 2 s window holds several cycles "
        "and gives an FFT resolution of 0.5 Hz, while still being short enough to follow changes of activity. Each window takes the label of its source recording; windows of the multi-activity clip "
        "take the label of the segment containing the window centre, and four windows overlapping the pick-up transition (10 to 12 s) are decoded but not scored.")]
S += [KeepTogether([img("raw_signals.png", 9.6), P("<b>Figure 1.</b> Raw accelerometer and gyroscope signals (10 s excerpts, 50 Hz after harmonisation) for the four activities. Still and Standing are almost flat; Walking is periodic; Jumping has large, sharp acceleration peaks.", cap)])]
S += [P("<b>Features.</b> Eight features per sensor (16 in total), computed per window: time domain, mean of the magnitude, variance, RMS, signal magnitude area (SMA) and mean absolute inter-axis correlation; "
        "frequency domain (FFT of the detrended, Hann-windowed magnitude), dominant frequency, spectral energy and normalised spectral entropy. "
        "Variance, RMS, SMA and energy differ by up to six orders of magnitude between Still and Jumping, so they are log10-transformed before normalisation. All features are then <b>Z-scored with a scaler fitted on training windows only</b> and applied unchanged to the test files; "
        "fitting on test data would leak the test distribution into training, and Z-scoring stops large-valued features from dominating the diagonal Gaussians. "
        f"Variance, RMS and SMA rise with movement intensity (median accelerometer variance {mf['Still']['acc_var']}, {mf['Standing']['acc_var']}, {mf['Walking']['acc_var']:.1f} and {mf['Jumping']['acc_var']:.0f} for Still, Standing, Walking and Jumping); "
        "dominant frequency separates rhythmic from noise-like motion; spectral entropy is high for broadband noise and lower for periodic motion; gyroscope SMA captures body rotation (Figure 2).")]
S += [KeepTogether([img("feature_boxplots.png", 12.2), P("<b>Figure 2.</b> Distribution of six features per activity (training windows). Variance and gyroscope SMA separate the activities by orders of magnitude; dominant frequency and entropy separate periodic from noise-like motion.", cap)])]
vt = [["Activity", "Video link"]] + [[a, PH("add your video link here: video not yet recorded or linked")] for a in ["Standing", "Walking", "Jumping", "Still"]]
S += [P("<b>Table 2.</b> Placeholder for the four activity videos (to be recorded and linked by the author).", tcap), tbl(vt, [3, 10.5], fs=7.8)]

# ---------------------------------------------------------------- 3
S += [P("HMM Setup and Implementation", h1),
      P("<b>Model.</b> The four hidden states are the activities. Each state emits the 16-dimensional Z-scored feature vector from a Gaussian with diagonal covariance (variance floor 0.01 in Z units, to prevent collapse on near-constant features such as Still). "
        "The parameters are the initial distribution <i>pi</i>, the 4 x 4 transition matrix <i>A</i> and the emission means and variances <i>B</i>. Everything is implemented in numpy (src/hmm.py, class GaussianHMM): "
        "forward and backward passes in log-space with log-sum-exp (no underflow), Viterbi decoding in log-space, and Baum-Welch over several sequences, with a transition floor of 0.001 as smoothing."),
      P(f"<b>Supervised vs unsupervised.</b> Because labels exist, <i>pi</i>, <i>A</i>, and the emission means and variances are first initialised from the labelled training sequences (supervised). Baum-Welch then re-estimates them from the observations alone, "
        f"ignoring the labels (unsupervised refinement). Training stops when the total log-likelihood improves by less than 1e-4 (cap of 200 iterations). From the supervised start it stopped after {R['bw_iters']} iterations "
        f"(log-likelihood {R['ll_first']:.1f}, final change {R['ll_change']:.0e}): the classes are so well separated that the labelled start is already a fixed point of EM. "
        f"To show that EM does converge on its own, an additional unsupervised run from random initialisations was made (5 restarts): {sum(1 for x in R['unsup_ll'] if x > 0)} restarts reached the same log-likelihood ({max(R['unsup_ll']):.1f}) within {R['unsup_iters']} iterations and "
        f"the other {sum(1 for x in R['unsup_ll'] if x < 0)} stalled in a much worse local optimum ({min(R['unsup_ll']):.1f}). After naming the learned states by majority vote on training windows, the best run gives the same held-out accuracy (Figure 6)."),
      P(f"<b>Training sequences.</b> No recording contains more than one activity, so a transition matrix estimated from single recordings would be an uninformative near-identity matrix. "
        f"Instead {R['n_seq']} training sequences ({R['seq_windows']} windows) were built by concatenating eight randomly ordered 10 s training chunks, each chunk of a different activity from the previous one (seeded). "
        "The off-diagonal transition probabilities therefore reflect this construction (a switch every 9 windows on average, destination chosen at random), not measured behaviour. "
        "Sanity tests (run in the notebook) confirm that the forward pass and Viterbi path match brute-force enumeration on a small model, that the rows of <i>A</i> sum to 1, and that the log-likelihood never decreases across EM iterations."),
      P("<b>Why Viterbi and Baum-Welch.</b> The task is to recover the most likely sequence of activities from noisy feature sequences, which is exactly what Viterbi decoding returns; "
        "it uses the transition matrix to favour persistent activities, so isolated noisy windows are less likely to flip the label, and this directly affects sensitivity (fewer missed windows of an activity) and specificity (fewer false alarms for other activities). "
        "Baum-Welch fits the parameters by maximising the likelihood of the observations, handles the fact that state assignments are only softly known, and its convergence criterion gives a principled stopping rule. "
        "Overall accuracy summarises how many windows Viterbi labels correctly, while one-vs-rest sensitivity and specificity show which activities are missed or confused.")]

# ---------------------------------------------------------------- 4
S += [P("Results and Interpretation", h1),
      P(f"All test recordings are decoded with Viterbi, each as its own sequence, and none was used for training, scaling or tuning (split by recording, not by window). "
        f"The two required unseen files (jumping_test_01 and still_test_01, iPhone 11 Pro) give {n_u} windows; they contain only Still and Jumping, so Walking and Standing have no test windows and their sensitivity is undefined (Table 3). "
        f"The continuous multi-activity clip, also held out, adds Standing and Walking (Table 4). Its segment boundaries were read from the signal energy and confirmed by the author, and the scored windows number only "
        f"{sum(v for k, v in wc['test_multi'].items() if k != 'transition')}, so per-class rates are rough estimates.")]
S += [P(f"<b>Table 3.</b> One-vs-rest metrics on the two required unseen files. Overall multi-class accuracy: {a_u:.3f} ({n_u} windows).", tcap), tbl(metrics_rows(R["table_unseen2"]), [3, 3.6, 2.6, 2.6, 4.2]),
      P(f"<b>Table 4.</b> One-vs-rest metrics on all held-out recordings (two unseen files plus the multi-activity clip, {n_h} windows). Overall accuracy: {a_h:.3f}. "
        f"Per file: jumping_test_01 {pf['jumping_test_01']:.3f}, still_test_01 {pf['still_test_01']:.3f}, multi_test_01 {pf['multi_test_01']:.3f}.", tcap), tbl(metrics_rows(R["table_heldout"]), [3, 3.6, 2.6, 2.6, 4.2])]
cmp_rows = [["Model", "2 unseen files", "All held-out", "Multi-activity only"]]
for m_ in ["Viterbi + Baum-Welch (main)", "Viterbi, supervised init only", "Frame-wise Gaussian (no transitions)"]:
    cmp_rows.append([m_, f"{acc[m_ + ' | 2 unseen files']:.3f}", f"{acc[m_ + ' | 2 unseen + multi-activity']:.3f}", f"{acc[m_ + ' | multi-activity only']:.3f}"])
cmp_rows.append(["Unsupervised Baum-Welch (random init)", f"{acc['Unsupervised Baum-Welch (random init) | 2 unseen files']:.3f}", f"{acc['Unsupervised Baum-Welch (random init) | 2 unseen + multi-activity']:.3f}", "-"])
S += [P("<b>Table 5.</b> Overall accuracy of the main model and of reference variants. The frame-wise classifier uses the same Gaussians but ignores transitions.", tcap), tbl(cmp_rows, [6.2, 2.9, 2.9, 3.4])]
S += [Spacer(1, 4), Table([[img("confusion_matrix_heldout.png", 6.3), img("transition_matrix.png", 6.3)],
        [P("<b>Figure 3.</b> Confusion matrix on all held-out windows (rows: true activity).", cap), P("<b>Figure 4.</b> Transition matrix A after Baum-Welch (annotated).", cap)]], colWidths=[8.4 * cm, 8.4 * cm])]
S += [KeepTogether([img("emission_means.png", 15.5), P("<b>Figure 5.</b> Emission means of the four states for the 16 Z-scored features. The states differ mainly in the variance, RMS, SMA and energy features, and Still and Standing differ only by a few tenths of a Z-unit in the low-energy features.", cap)])]
S += [KeepTogether([img("loglik_convergence.png", 13.2), P("<b>Figure 6.</b> Baum-Welch log-likelihood. Left: supervised initialisation, already converged. Right: unsupervised EM from five random initialisations; it increases monotonically and stops when the change falls below 1e-4.", cap)])]
S += [KeepTogether([img("decoded_vs_true.png", 13.6), P("<b>Figure 7.</b> Viterbi-decoded versus true activity for the three held-out recordings (the gap in the multi-activity line is the unscored pick-up transition).", cap)])]

# ---------------------------------------------------------------- 5
S += [P("Discussion and Conclusion", h1),
      P(f"<b>Easiest and hardest activities.</b> Jumping in the unseen jumping file was decoded perfectly ({R['cm_unseen2'][3][3]} of {sum(R['cm_unseen2'][3])} windows), and Still in the unseen still file almost perfectly "
        f"({R['cm_unseen2'][0][0]} of {sum(R['cm_unseen2'][0])}); Walking and Standing in the multi-activity clip were all correct, though from only 6 and 8 windows. The misses are informative. "
        f"The five Still windows decoded as Standing are consecutive (13 to 18 s of the recording): that stretch has a short burst of vibration with accelerometer variance up to 0.028, above any training Still window (maximum 0.0046) and within the Standing range, "
        "so this is arguably a genuine disturbance in a clip labelled Still rather than a random model error. It does show that Still versus Standing is the thinnest margin: both are low-energy and noise-like, "
        "and only the magnitude of tiny movements separates them. The largest failure is Jumping in the multi-activity clip: only 3 of its 11 windows were recognised and 8 were decoded as Walking. "
        f"The jumps there are far gentler than in training (median accelerometer variance {jv['multi_test_01']} against {jv['jumping_train_01']}, {jv['jumping_train_02']} and {jv['jumping_test_01']} in the other jumping recordings; gyroscope SMA {jg['multi_test_01']} against about {jg['jumping_train_01']}), "
        "so they fall between the training Walking and Jumping distributions. This is a behaviour or placement shift; the model was not adjusted after seeing it."),
      P(f"<b>Transition probabilities.</b> The learned matrix has a diagonal of {A.diagonal().min():.2f} to {A.diagonal().max():.2f} and off-diagonal entries of {np.min(A[~np.eye(4, dtype=bool)]):.2f} to {np.max(A[~np.eye(4, dtype=bool)]):.2f}. "
        f"The high diagonal is realistic (an expected stay of about {dwell:.0f} windows, i.e. {dwell:.0f} s at a 1 s window step, matching the persistence of activities). "
        "However, the near-uniform off-diagonal entries are an artefact of the training sequences (random chunk order), so they should not be read as real preferences such as Walking leading to Jumping more often than Still does. "
        "Real transition structure would need continuous recordings of natural behaviour."),
      P(f"<b>Noise and sampling rate.</b> The unseen clips come from a different phone, and its still recording has a higher noise floor (median window variance 0.0006 against 0.0002 in training), yet this stays far below Standing (median {mf['Standing']['acc_var']}), "
        "so the 50 Hz harmonisation and the log-scaled features absorbed the device difference. For Still windows the dominant-frequency feature is driven by noise and carries little information, which the large Still variance in that dimension reflects. "
        "At 50 Hz and a 2 s window the frequency resolution is 0.5 Hz, which is coarse for the 1 to 2 Hz walking cadence."),
      P(f"<b>Does it generalise?</b> Only partly, and the evidence is limited. The pooled accuracy ({a_h:.3f}) is dominated by the long Still recording; the held-out set is small, windows overlap by 50%, and the Walking, Standing and multi-activity results come from a single short clip. "
        f"The main HMM did not beat a frame-wise Gaussian classifier that ignores transitions ({fw_h:.3f} against {a_h:.3f} overall, {fw_m:.3f} against {a_m:.3f} on the multi-activity clip, a difference of one window), which is expected because the test recordings are almost entirely single-activity and the transition matrix is constructed rather than learned. "
        "All recordings come from one project on a single day with two phones, and the multi-activity boundaries were chosen from the same signal energy that the features measure, which can flatter that clip's score."),
      P("<b>Improvements.</b> (1) More data: several sessions, participants, phone placements and jump intensities, with continuous recordings of natural activity sequences to estimate <i>A</i> from real behaviour. "
        "(2) Better features: per-axis and jerk statistics, autocorrelation or wavelet features, and a longer window or per-user normalisation for the low-frequency Still/Standing contrast. "
        "(3) Additional sensors such as the magnetometer, barometer or gravity direction to distinguish Standing from Still by orientation. "
        "(4) Model variants: Gaussian-mixture emissions for activities with several intensity levels, full-covariance emissions, hidden semi-Markov models for explicit duration, or a hierarchical HMM."),
      P(f"<b>Conclusion.</b> A from-scratch Gaussian HMM with supervised initialisation and Baum-Welch refinement recovers the four activities with {a_u:.1%} accuracy on the two required unseen recordings and {a_h:.1%} on all held-out data. "
        "The results are encouraging for clearly different activities, but jump intensity, the Still/Standing margin and the small, single-project test set limit what can be claimed about generalisation.")]

def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 7.5); c.drawCentredString(A4[0] / 2, 1.0 * cm, f"Page {d.page}"); c.restoreState()
doc = SimpleDocTemplate(str(REPO / "report/report.pdf"), pagesize=A4, leftMargin=1.9 * cm, rightMargin=1.9 * cm, topMargin=1.6 * cm, bottomMargin=1.7 * cm,
                        title="Modeling Human Activity States Using Hidden Markov Models")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("built")
