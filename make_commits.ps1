# make_commits.ps1
# Rebuilds this repo's history as many small commits, all authored as Joellate.
# Run from the repo root (the folder that contains .git):
#   powershell -ExecutionPolicy Bypass -File .\make_commits.ps1
# Files on disk are not changed; only the (unpushed) local history is replaced.

if (-not (Test-Path .git)) { throw "Run this from the repo root (the folder that contains .git)." }
$email = Read-Host "Your GitHub email (use your noreply address)"
git config user.name "Joellate"
git config user.email $email

# start a fresh history; files stay on disk, only the index is cleared
git checkout --orphan fresh-main
git rm -r --cached . -q --ignore-unmatch

function Commit-Files([string]$Message, [string[]]$Paths) {
    foreach ($p in $Paths) { git add -- $p 2>$null }
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) { git commit -q -m $Message; Write-Host "committed: $Message" }
}

# setup
Commit-Files "add gitignore" @(".gitignore")
Commit-Files "add requirements" @("requirements.txt")

# raw recordings, one commit each
Commit-Files "add walking training clip"            @("data/raw/walking_train_01_*")
Commit-Files "add first jumping clip"               @("data/raw/jumping_train_01_*")
Commit-Files "add second jumping clip"              @("data/raw/jumping_train_02_*")
Commit-Files "add standing clip"                    @("data/raw/standing_train_01_*")
Commit-Files "add still clip with gravity channel"  @("data/raw/still_train_01_*")
Commit-Files "add unseen jumping test clip"         @("data/raw/jumping_test_01_*")
Commit-Files "add unseen still test clip"           @("data/raw/still_test_01_*")
Commit-Files "add multi-activity clip"              @("data/raw/multi_test_01_*")
Commit-Files "add recordings manifest"              @("data/recordings_manifest.csv")

# code
Commit-Files "add loading and resampling helpers"           @("src/data_utils.py")
Commit-Files "add windowing and feature extraction"         @("src/features.py")
Commit-Files "add gaussian hmm with viterbi and baum-welch" @("src/hmm.py")

# processed data, one commit each
foreach ($r in "walking_train_01","jumping_train_01","jumping_train_02","standing_train_01","still_train_01","jumping_test_01","still_test_01","multi_test_01") {
    Commit-Files "add 50 hz data for $r" @("data/processed/${r}_50hz.csv")
}

# derived tables
Commit-Files "add data quality table"       @("data/qc_table.csv")
Commit-Files "add chunk table"              @("data/chunk_table.csv")
Commit-Files "add window features"          @("data/features_raw.csv")
Commit-Files "add z-scored features"        @("data/features_zscored.csv")
Commit-Files "add trained hmm parameters"   @("data/hmm_params.npz")

# notebook and figures
Commit-Files "add analysis notebook"             @("notebooks/hmm_activity_recognition.ipynb")
Commit-Files "add raw signal plot"               @("figures/raw_signals.png")
Commit-Files "add multi-activity overview plot"  @("figures/multi_activity_overview.png")
Commit-Files "add feature boxplots"              @("figures/feature_boxplots.png")
Commit-Files "add feature mean heatmap"          @("figures/feature_means_heatmap.png")
Commit-Files "add transition matrix plot"        @("figures/transition_matrix.png")
Commit-Files "add emission means plot"           @("figures/emission_means.png")
Commit-Files "add log-likelihood plot"           @("figures/loglik_convergence.png")
Commit-Files "add confusion matrix for unseen files"        @("figures/confusion_matrix_unseen2.png")
Commit-Files "add confusion matrix for all held-out data"   @("figures/confusion_matrix_heldout.png")
Commit-Files "add decoded vs true plot"          @("figures/decoded_vs_true.png")
Commit-Files "add metrics tables"                @("figures/metrics_unseen2.csv","figures/metrics_all_heldout.csv")
Commit-Files "add model comparison table"        @("figures/model_comparison.csv")
Commit-Files "add results summary json"          @("figures/results.json")

# report and readme
Commit-Files "add report build script"  @("report/build_report.py")
Commit-Files "add report pdf"           @("report/report.pdf")
Commit-Files "add readme"               @("README.md")

# catch anything not listed above
git add -A
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) { git commit -q -m "add remaining files" }

# make this the main branch and check the result
git branch -M main
if (-not (git remote)) { git remote add origin https://github.com/Joellate/HMM.git }
Write-Host ""
Write-Host "Commits:" (git rev-list --count HEAD)
git log --format="%h %an <%ae> | %s"
Write-Host ""
Write-Host "Co-author lines (should print nothing):"
git log --format=%B | Select-String -Pattern "co-authored"
Write-Host ""
Write-Host "Uncommitted changes (should print nothing):"
git status --short
Write-Host ""
Write-Host "Next: git push -u origin main   (sign in with gh auth login or a personal access token, not your password)"
