# Workflow

Run commands from the repository root with authorized local datasets.

## 1. Participant identities and splits

The manifest must declare `identity_scope: "global_participant"`, nonempty
`train_subjects`, `val_subjects`, `test_subjects`, and optional
`external_subjects`. Keep every session of a person together.
`participant_map` maps local PKL identifiers to globally unique people.
Labeler CSV identifiers use `dataset:group_id`; session labeling uses
`--dataset:directory_name`. Register target participants before labeling.
A synthetic schema is in `examples/synthetic_split_manifest.json`.

## 2. Logistic fitting and FIX demonstrations

```bash
python labeling/fit_logistic_labeler.py --aligned-labeled-csv LOCAL/aligned.csv --split-manifest LOCAL/splits.json --output outputs/labeler/model.json
python labeling/prepare_reference_context.py --aligned-labeled-csv LOCAL/aligned.csv --logistic-model outputs/labeler/model.json --split-manifest LOCAL/splits.json --reference-dataset FIX --source-description "ACTUAL PUBLIC FIX RELEASE" --output outputs/labeler/references.json
```

Use the actual dataset value from the FIX source table. Demonstrations come only
from approved training participants. The default signal threshold 0.39 is
predeclared. Labels use signal=1 during labeling and noise=1 downstream.

## 3. Grounded labeling

```bash
python labeling/grounded_labeler.py --reference-context outputs/labeler/references.json --split-manifest LOCAL/splits.json --logistic-model outputs/labeler/model.json --ica-root LOCAL/session_features --dataset HCP --output-root outputs/labels --mode selective --model ACTUAL_SERVICE_ID --api-base https://YOUR_AUTHORIZED_ENDPOINT/v1 --allow-feature-upload
```

Use `--mode batch` to review all ICs. Set `SLICE_ICA_API_KEY` in the environment.
The upload flag acknowledges applicable transmission permission. HTTPS is
required. Every request includes training-only FIX demonstrations, but no target
true label or participant identifier. Outputs preserve the [0.35,0.65] routing
interval, override or 0.8/0.2 fusion, probabilities, and FIX noise-index lists.

Use a fresh output directory. Feature rows must follow contiguous MELODIC IC
order. An available `ic_index` is checked; unindexed source tables must already
have their ordering verified.

## 4. Trusted PKL preparation

Convert authorized ICA inputs using `classification/data2pkl_script/data2pkl.py`.
Then match descriptors and noise targets to the grounded run:

```bash
python labeling/attach_pseudo_provenance.py --pkl-root LOCAL/matching_pseudo_pkls --label-run outputs/labels --output-root outputs/pseudo_cache
```

This checks subject/session/IC identity, descriptor values and label convention.
Never load PKLs from untrusted sources.

## 5. Training

Place the global-ID manifest at `outputs/training/audit/split_manifest.json`.
On Linux, configure:

```bash
export HCP_PROJECT_ROOT="$PWD/classification"
export HCP_TRUE_DIR="/path/to/authorized/expert_pkls"
export HCP_PSEUDO_DIR="$PWD/outputs/pseudo_cache"
export HCP_EXP_DIR="$PWD/outputs/training"
export HCP_PARAMS_JSON="$PWD/configs/hybrid_recipe.json"
export HCP_ABLATION=full
export HCP_SEED=42
python classification/scripts/run_hybrid_ablation_trial.py
```

Training inputs are participant-disjoint from validation and test. The default
five OOF folds keep all sessions together and use fixed epoch budgets. Pseudo
records enter an OOF fit only when their upstream labeler/reference participants
also exclude that fold; unknown provenance is rejected. The final encoder uses
training data and validation-only checkpoint selection. The booster is fitted
on expert OOF rows using probability, logit, and 185 FIX descriptors.

## 6. Frozen evaluation

```bash
python classification/scripts/infer_frozen.py --artifact outputs/training/audit/ablation_artifact.json --split-manifest LOCAL/splits.json --pkl-root LOCAL/heldout_pkls --gold-labels-confirmed --output outputs/predictions.json
python tools/evaluate_frozen_predictions.py --input MSC=outputs/msc_predictions.json HCP=outputs/hcp_predictions.json --output outputs/external_metrics.json
```

Use `--gold-labels-confirmed` only for expert gold, not automatic labels. Otherwise
the gold field is null and metric calculation is rejected. Pooling requires the
same artifact hash and validation-selected threshold, and rejects duplicate ICs.
No test label is used for fitting or threshold optimization.

## 7. Analysis and checks

`analysis/` contains group-audit and QC/FC utilities. Participant-cluster sign-flip
tests and BH adjustment are available in `analysis/cluster_statistics.py`.
Best-pair and worst-pair DI are separately named. Supplied manuscript summaries
remain distinct from any newly computed analysis outputs.

```bash
python tools/release_checks.py
python -m unittest discover -s tests -v
```

The tests use synthetic inputs and mocked callbacks; they do not run experiments.
