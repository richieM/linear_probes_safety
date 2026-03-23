"""
train_probes.py
───────────────
Load cached activations, train one logistic-regression probe per transformer
layer, evaluate AUROC on the held-out test split, and compare to the
behavioral baseline.

Outputs (all written to results/):
  auroc_by_layer.png      — line chart of probe AUROC at each layer
  comparison_table.csv    — best probe vs baseline vs random
  findings.md             — auto-generated writeup paragraph
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def load_cached(path: str = "results/activations.npz"):
    data = np.load(path, allow_pickle=True)
    return (
        data["activations"],       # [n, n_layers, hidden_dim]
        data["labels"],            # [n]
        data["splits"],            # [n]  dtype object/str
        data["baseline_scores"],   # [n]
    )


def train_probes(activations, labels, splits):
    """
    Train one logistic regression per layer on the train split.
    Evaluate AUROC on the test split.

    Returns
    -------
    auroc_per_layer : list of float, length = n_layers
    """
    train_mask = splits == "train"
    test_mask  = splits == "test"

    n_layers = activations.shape[1]
    auroc_per_layer = []

    for layer_idx in range(n_layers):
        X_train = activations[train_mask, layer_idx, :]  # [n_train, hidden]
        X_test  = activations[test_mask,  layer_idx, :]  # [n_test,  hidden]
        y_train = labels[train_mask]
        y_test  = labels[test_mask]

        # Standardize: zero mean, unit variance per feature.
        # Fit on train only — transform both.
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # L2 logistic regression. C=1.0 is a reasonable default;
        # max_iter=1000 ensures convergence on high-dim data.
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        clf.fit(X_train, y_train)

        # AUROC uses probability scores, not hard predictions.
        y_scores = clf.predict_proba(X_test)[:, 1]  # P(harmful)
        auroc = roc_auc_score(y_test, y_scores)
        auroc_per_layer.append(auroc)

        print(f"  Layer {layer_idx:2d}  AUROC = {auroc:.4f}")

    return auroc_per_layer


def evaluate_baseline(baseline_scores, labels, splits):
    """
    Compute AUROC for the behavioral baseline (model's own P(YES) scores).
    """
    test_mask = splits == "test"
    return roc_auc_score(labels[test_mask], baseline_scores[test_mask])


def plot_auroc_by_layer(auroc_per_layer, baseline_auroc, output_path):
    """
    Line chart: probe AUROC at each layer vs a horizontal line for the baseline.
    """
    n_layers = len(auroc_per_layer)
    layers = list(range(n_layers))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(layers, auroc_per_layer, marker="o", linewidth=2,
            color="#2563eb", label="Linear probe")
    ax.axhline(baseline_auroc, color="#dc2626", linewidth=1.5,
               linestyle="--", label=f"Behavioral baseline ({baseline_auroc:.3f})")
    ax.axhline(0.5, color="#9ca3af", linewidth=1,
               linestyle=":", label="Random (0.500)")

    best_layer = int(np.argmax(auroc_per_layer))
    ax.annotate(
        f"Best: layer {best_layer}\nAUROC {auroc_per_layer[best_layer]:.3f}",
        xy=(best_layer, auroc_per_layer[best_layer]),
        xytext=(best_layer + 1.5, auroc_per_layer[best_layer] - 0.06),
        arrowprops=dict(arrowstyle="->", color="#374151"),
        fontsize=9,
    )

    ax.set_xlabel("Transformer layer", fontsize=12)
    ax.set_ylabel("AUROC (test split)", fontsize=12)
    ax.set_title("Linear probe AUROC by layer — harmful query detection", fontsize=13)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(0.45, 1.02)
    ax.set_xticks(layers)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {output_path}")


def write_comparison_table(auroc_per_layer, baseline_auroc, output_path):
    best_layer = int(np.argmax(auroc_per_layer))
    df = pd.DataFrame([
        {"Method":               "Best linear probe",
         "Layer":                best_layer,
         "AUROC":                f"{auroc_per_layer[best_layer]:.4f}"},
        {"Method":               "Behavioral baseline (self-classification)",
         "Layer":                "—",
         "AUROC":                f"{baseline_auroc:.4f}"},
        {"Method":               "Random",
         "Layer":                "—",
         "AUROC":                "0.5000"},
    ])
    df.to_csv(output_path, index=False)
    print(f"Comparison table saved → {output_path}")
    print(df.to_string(index=False))
    return df


def write_findings(auroc_per_layer, baseline_auroc, output_path):
    best_layer = int(np.argmax(auroc_per_layer))
    best_auroc = auroc_per_layer[best_layer]
    n_layers   = len(auroc_per_layer)

    # Find where probes first exceed the baseline
    exceeds = [i for i, a in enumerate(auroc_per_layer) if a > baseline_auroc]
    first_exceeds = exceeds[0] if exceeds else None

    # Find early-layer performance
    early_auroc = np.mean(auroc_per_layer[:3])
    late_auroc  = np.mean(auroc_per_layer[-3:])

    findings = f"""# Linear Probe Findings

## Setup
- Model: Gemma-2-2B ({n_layers} transformer layers, 2304 hidden dim)
- Dataset: 400 examples (200 harmful / 200 benign), 80/20 train-test split
- Probe: L2 logistic regression, standardized activations, last-token position
- Baseline: model self-classification via P(YES) from next-token logits

## Results

**Best probe layer:** {best_layer} — AUROC {best_auroc:.4f}
**Behavioral baseline AUROC:** {baseline_auroc:.4f}
**Random baseline:** 0.5000

## Observations

Early layers (0–2) achieve a mean AUROC of {early_auroc:.3f}, suggesting \
{"the model begins encoding harm signals even in early layers." if early_auroc > 0.65
 else "limited harm signal in early representations — the concept hasn't been built yet."}

The best performance at layer {best_layer} (AUROC {best_auroc:.4f}) \
{"substantially" if best_auroc - baseline_auroc > 0.05 else "slightly"} \
{"exceeds" if best_auroc > baseline_auroc else "falls short of"} \
the behavioral baseline ({baseline_auroc:.4f}).

{f"Probes first exceed the baseline at layer {first_exceeds}, suggesting harm representations become linearly separable before the model's output behaviour fully reflects them."
  if first_exceeds is not None else
  "Probes did not consistently exceed the behavioral baseline, suggesting the model's self-classification is quite strong."}

Late layers (last 3) average AUROC {late_auroc:.3f}, \
{"showing the signal is maintained through the final layers."
  if late_auroc > 0.75 else
  "a slight drop from peak, possibly as the model prepares token-generation representations."}

## Key Takeaway

{"The probe outperforms the model's own safety judgement — meaning harmful intent is encoded in the internal representations even in cases where the behavioral output might not flag it."
  if best_auroc > baseline_auroc else
  "The behavioral baseline matches or exceeds the best probe here. This could reflect a strong alignment between the model's safety training and its internal representations, or that the test split has limited hard cases."}
"""

    Path(output_path).write_text(findings)
    print(f"Findings written → {output_path}")
    return findings


def run_training(
    activations_path: str = "results/activations.npz",
    output_dir: str = "results",
):
    Path(output_dir).mkdir(exist_ok=True)

    print("Loading cached activations...")
    activations, labels, splits, baseline_scores = load_cached(activations_path)
    print(f"  Activations: {activations.shape}  "
          f"(train {(splits=='train').sum()}, test {(splits=='test').sum()})")

    print("\nTraining probes (one per layer)...")
    auroc_per_layer = train_probes(activations, labels, splits)

    print("\nEvaluating behavioral baseline...")
    baseline_auroc = evaluate_baseline(baseline_scores, labels, splits)
    print(f"  Baseline AUROC = {baseline_auroc:.4f}")

    plot_auroc_by_layer(
        auroc_per_layer, baseline_auroc,
        output_path=f"{output_dir}/auroc_by_layer.png",
    )
    write_comparison_table(
        auroc_per_layer, baseline_auroc,
        output_path=f"{output_dir}/comparison_table.csv",
    )
    write_findings(
        auroc_per_layer, baseline_auroc,
        output_path=f"{output_dir}/findings.md",
    )

    return auroc_per_layer, baseline_auroc


if __name__ == "__main__":
    run_training()
