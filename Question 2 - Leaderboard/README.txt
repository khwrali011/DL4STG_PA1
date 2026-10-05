QUESTION 2 — COMPREHENSIVE AUTOFORMER SEARCH

Place question2_comprehensive.py in "Question 2 - Leaderboard", beside Data.
Keep your existing CUDA-enabled PyTorch environment. Do not reinstall a CPU
PyTorch build over it. Additional dependencies, if needed:
  python -m pip install numpy pandas matplotlib

Recommended command (PowerShell, activated environment):
  python .\question2_comprehensive.py --data-dir .\Data --device cuda --min-train-minutes 60 --max-search-minutes 90

Optional architecture checks:
  python .\question2_comprehensive.py --self-test
Optional tiny pipeline check (creates a separate _smoke folder, no predictions
for submission; its results are not evidence of forecasting quality):
  python .\question2_comprehensive.py --data-dir .\Data --device cuda --smoke

RUNTIME AND RESUMING
The default requires at least 60 cumulative minutes spent in completed model
training and early-stop evaluation. If the original shortlist finishes faster,
additional distinct configurations are evaluated on three folds and three
seeds. No sleep is used. Checkpoint writing and other overhead are extra.
90 minutes is a soft search ceiling: a configuration/finalist is completed as
a comparable unit. The minimum workload, audit, final fits and ablation can
make total runtime exceed 90 minutes, potentially by a substantial amount.
There is no promise that every run will improve the hidden-test score.
On the user's P1000 this is a substantially larger workload than the prior
nine-minute script. Do not interpret elapsed training time as model quality.
Completed work counts on resume, so restarting a finished search need not
consume another hour.

Rerun the exact command to resume. Every completed epoch saves model, optimizer,
RNG states, shuffle-generator state and best checkpoint. An interrupted epoch
restarts from the last completed epoch; partial epoch work may be repeated.
Changing data/code/statistical settings requires a fresh --out folder; changing
the minimum/maximum training budget is allowed. Resume on the same device for
closest reproducibility. Checkpoints must be your own, trusted local files.

WHAT IS DIFFERENT
* Reference-style progressive decomposition, seasonal-specific final layer
  normalization, circular convolutional embeddings and accumulated decoder trend.
* FFT delay discovery and k = factor*log(length), with batch-shared delay
  selection while training, per-example while evaluating. Both reference-like
  unscaled correlation and normalized correlation are candidate choices.
* Raw-unit standardization, log targets with raw MSE, and mixed raw/log loss.
  Direct raw-unit training targets the original-unit RMSE more directly.
* Contexts 336/672/1008, widths 32/48/64, one/two encoder layers, kernels
  7/25/49, optional-data subsets, a learned covariate residual head, recent
  training spans, recency-weighted loss, learning rate and dropout variants.
  The covariate head is an explicit experimental addition to Autoformer.
* Finalists compared across three seeds. Single-seed screening only narrows
  the search and is not reported as evidence that a design is superior.
* Single Autoformer, three-seed ensemble, and two-configuration ensemble.
  Every ensemble member remains an Autoformer. Costs sum over its members.
* Optional bounded affine bias adjustment. It is enabled only when fitting
  on earlier development folds improves forecasts on later folds by >=2%,
  without worsening either evaluated fold by more than 3%. This is an
  experimental safeguard, not the analytic log-normal correction in the textbook.
* A matched with/without-optional-data ablation across all three seeds.

CHRONOLOGY / INFORMATION BOUNDARY
Actual training has 43,656 target observations. Test forecasts are exactly
168 values for time_idx 43657 through 43824. Test targets must be blank.
Known future optional variables are permitted by the assignment and fed into
the decoder; no future target is used as an input or a training label.
Scalers are fitted only to each fit prefix, or the configured recent portion
of that prefix. Optional binary indicators retain their 0/1 representation.
The six continuous variables are standardized; D/E/F first receive log1p.

Three development folds (inclusive, one-based positions):
  Fold 1: fit 1–30216; early-stop 30217–30888; development 30889–32904.
  Fold 2: fit 1–35592; early-stop 35593–36264; development 36265–38280.
  Fold 3: fit 1–39624; early-stop 39625–40296; development 40297–42312.
Each development region contains twelve non-overlapping 168-step blocks.
Each origin may use all target observations preceding it as context, even
when the fixed model was fitted earlier. Early-stop targets never fit weights.
The best epoch is selected on early-stop blocks; development targets choose
configurations and ensemble membership. The final schedule is the median best
epoch across development folds, separately for each selected seed.

The final eight blocks 42313–43656 form an audit of the frozen pipeline:
no ensemble weights, calibration or epochs are changed after viewing it.
Parts of this audit region appeared in the user's previous experiments, so
it is held out from this search but is not a globally untouched dataset.
Final models are fitted afresh on all available observed training data (or
the chosen recent span), for the fixed selected number of epochs.

Selection criterion: 0.75*pooled development RMSE + 0.25*worst-fold RMSE.
Prefer the option with fewer parameters if it lies within 1% of the minimum.
This criterion is declared before the search; it does not use leaderboard
scores, unknown test targets, or the audit targets.

OUTPUTS: question2_comprehensive_output/
  leaderboard_values.txt   Exactly 168 comma-separated test predictions.
  forecast.csv             Predictions with time_idx for inspection.
  declaration.json         Reproducible P and selected-pipeline E counts.
  confirmation_report.json Frozen-pipeline audit metrics, error by horizon,
                           high-target errors, and diagnostic baselines.
  candidate_ranking.json   Single-model and ensemble development comparisons.
  refined_summary.json    All finalist metrics across folds and seeds.
  screening.json          Exploratory single-seed screening results.
  frozen_selection.json   Choices recorded before audit evaluation.
  optional_ablation.json  Matched comparisons across seeds.
  audit_predictions.csv   All audit actual/prediction pairs.
  audit_forecasts.png / test_forecast.png
  cache/                  Epoch checkpoints, final weights and predictions.

EPOCH AND PARAMETER DECLARATIONS
The assignment says E counts actual early-stop training plus refitting; an
ensemble pays the summed epochs and parameters of its members. The script
separately records selected-lineage work and all other experiments:
  parameters_P_for_form: sum of final-member trainable parameters; also counts
  two fitted affine coefficients if nonidentity calibration is applied.
  epochs_E_selected_pipeline: all selected members' actual three-fold
  early-stopping epochs plus their audit fits plus their final fits.
  final_fit_epochs_sum: final fits only (not the full declared pipeline).
  search_and_ablation_epochs_total: all experiments, for transparency.
Unrelated discarded search models and optional ablation models do not produce
the submitted predictions and are reported separately. This selected-pipeline
count is conservative, including audit refits. Do not reuse P=22210 or E=27;
the new final model/ensemble can have different values.

RESEARCH SOURCES / ATTRIBUTION
Wu et al. (2021), Autoformer: Decomposition Transformers with Auto-Correlation
for Long-Term Series Forecasting, NeurIPS.
  https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html
Authors' Autoformer implementation inspected for decomposition, normalization,
decoder initialization and autocorrelation semantics:
  https://github.com/thuml/Autoformer/blob/main/models/Autoformer.py
  https://github.com/thuml/Autoformer/blob/main/layers/Autoformer_EncDec.py
  https://github.com/thuml/Autoformer/blob/main/layers/AutoCorrelation.py
This script is a compact, independently written adaptation of the mechanisms,
not a claim to exactly reproduce the authors' full configuration. Differences
include embedding external covariates directly, recent-state cross mixing,
optional normalized correlation, covariate head and loss/search protocol.
Hyndman & Athanasopoulos, Forecasting: Principles and Practice, third edition:
  https://otexts.com/fpp3/tscv.html
  https://otexts.com/fpp3/ftransformations.html
These motivate chronological multi-step evaluation and care with means after
back-transformation; the bounded affine calibration here is a separate test.
DLinear/linear forecasting paper was also consulted as a comparative check:
  https://ojs.aaai.org/index.php/AAAI/article/view/26317
No DLinear, tree model or other replacement architecture produces submissions,
because the assignment requires Autoformer.

VALIDATION OF THIS DELIVERABLE
Forward/backward checks cover raw/log targets, contexts 336/672/1008 and
train/evaluation FFT paths. Decomposition reconstruction, scaler isolation,
time-split boundaries, a real-CSV smoke pipeline, export shape, checkpoint
resume and selected-lineage accounting are checked during development.
These checks are not a full training run or evidence of improved test accuracy.
Keep generative-AI assistance and subsequent edits in the assignment report.
