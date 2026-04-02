import os
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, top_k_accuracy_score,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  CONFIG — adjust these paths if needed
# ─────────────────────────────────────────────
MODEL_PATH   = "Model/keras_model.h5"
DATA_FOLDER  = "datas"          # your dataset root folder
IMG_SIZE     = 224              # model input size
OFFSET       = 20
WHITE_SIZE   = 300

labels = ['0','1','2','3','4','5','6','7','8','9',
          'a','b','c','d','e','f','g','h','i','j',
          'k','l','m','n','o','p','q','r','s','t',
          'u','v','w','x','y','z','delete','space']

print("=" * 60)
print("       SignSpeak — Model Evaluation Report")
print("=" * 60)

# ─────────────────────────────────────────────
#  1. LOAD MODEL
# ─────────────────────────────────────────────
print("\n[1] Loading model...")
model = load_model(MODEL_PATH)
model.summary()

# ─────────────────────────────────────────────
#  2. COUNT PARAMETERS
# ─────────────────────────────────────────────
total_params     = model.count_params()
trainable_params = sum([
    np.prod(w.shape) for w in model.trainable_weights
])
non_trainable    = total_params - trainable_params

print("\n" + "─" * 60)
print("  MODEL PARAMETERS")
print("─" * 60)
print(f"  Total Parameters      : {total_params:,}")
print(f"  Trainable Parameters  : {trainable_params:,}")
print(f"  Non-Trainable Params  : {non_trainable:,}")

# ─────────────────────────────────────────────
#  3. LOAD & PREPROCESS DATASET
# ─────────────────────────────────────────────
print("\n[2] Loading dataset from:", DATA_FOLDER)

X, y_true = [], []
class_counts = {}

for label in labels:
    folder = os.path.join(DATA_FOLDER, label)
    if not os.path.exists(folder):
        print(f"  ⚠ Folder not found: {folder} — skipping")
        continue

    files = [f for f in os.listdir(folder)
             if f.lower().endswith(('.jpg','.jpeg','.png'))]
    class_counts[label] = len(files)
    label_idx = labels.index(label)

    for fname in files:
        img_path = os.path.join(folder, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Preprocess same way as inference
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm    = img_rgb.astype(np.float32) / 255.0

        X.append(img_norm)
        y_true.append(label_idx)

X      = np.array(X)
y_true = np.array(y_true)

print(f"  Total images loaded   : {len(X)}")
print(f"  Number of classes     : {len(set(y_true))}")

# ─────────────────────────────────────────────
#  4. RUN PREDICTIONS
# ─────────────────────────────────────────────
print("\n[3] Running predictions (this may take a moment)...")
BATCH = 32
y_prob = []
for i in range(0, len(X), BATCH):
    batch      = X[i:i+BATCH]
    preds      = model.predict(batch, verbose=0)
    y_prob.extend(preds)
    print(f"  Processed {min(i+BATCH, len(X))}/{len(X)} images", end='\r')

y_prob  = np.array(y_prob)
y_pred  = np.argmax(y_prob, axis=1)
print(f"\n  Done.")

# ─────────────────────────────────────────────
#  5. CORE METRICS
# ─────────────────────────────────────────────
acc     = accuracy_score(y_true, y_pred)
top3    = top_k_accuracy_score(y_true, y_prob, k=3)
top5    = top_k_accuracy_score(y_true, y_prob, k=5)

# Per-class confidence stats
per_class_conf = {}
for i, lbl in enumerate(labels):
    mask = y_true == i
    if mask.sum() > 0:
        per_class_conf[lbl] = {
            'mean_conf' : y_prob[mask, i].mean(),
            'min_conf'  : y_prob[mask, i].min(),
            'max_conf'  : y_prob[mask, i].max(),
        }

print("\n" + "─" * 60)
print("  OVERALL METRICS")
print("─" * 60)
print(f"  Top-1 Accuracy        : {acc*100:.2f}%")
print(f"  Top-3 Accuracy        : {top3*100:.2f}%")
print(f"  Top-5 Accuracy        : {top5*100:.2f}%")

# ─────────────────────────────────────────────
#  6. CLASSIFICATION REPORT
# ─────────────────────────────────────────────
present_labels = sorted(set(y_true))
present_names  = [labels[i] for i in present_labels]

print("\n" + "─" * 60)
print("  CLASSIFICATION REPORT (Precision / Recall / F1)")
print("─" * 60)
report = classification_report(
    y_true, y_pred,
    labels=present_labels,
    target_names=present_names,
    digits=4
)
print(report)

# ─────────────────────────────────────────────
#  7. PER-CLASS CONFIDENCE TABLE
# ─────────────────────────────────────────────
print("─" * 60)
print("  PER-CLASS CONFIDENCE STATS")
print("─" * 60)
print(f"  {'CLASS':<10} {'SAMPLES':>8}  {'MEAN CONF':>10}  {'MIN CONF':>9}  {'MAX CONF':>9}")
print(f"  {'-'*10} {'-'*8}  {'-'*10}  {'-'*9}  {'-'*9}")
for lbl in labels:
    if lbl in per_class_conf:
        s = per_class_conf[lbl]
        c = class_counts.get(lbl, 0)
        flag = " ⚠" if s['mean_conf'] < 0.80 else ""
        print(f"  {lbl:<10} {c:>8}  {s['mean_conf']:>10.4f}  "
              f"{s['min_conf']:>9.4f}  {s['max_conf']:>9.4f}{flag}")

# ─────────────────────────────────────────────
#  8. WEAKEST CLASSES
# ─────────────────────────────────────────────
cm        = confusion_matrix(y_true, y_pred, labels=present_labels)
per_class_acc = cm.diagonal() / cm.sum(axis=1)

weak = sorted(
    [(present_names[i], per_class_acc[i]) for i in range(len(present_names))],
    key=lambda x: x[1]
)[:10]

print("\n" + "─" * 60)
print("  TOP 10 WEAKEST CLASSES (needs more data)")
print("─" * 60)
for rank, (cls, ac) in enumerate(weak, 1):
    bar = "█" * int(ac * 20) + "░" * (20 - int(ac * 20))
    print(f"  {rank:>2}. {cls:<10} [{bar}] {ac*100:.1f}%")

# ─────────────────────────────────────────────
#  9. PLOTS
# ─────────────────────────────────────────────
print("\n[4] Generating plots...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("SignSpeak — Model Evaluation Dashboard", fontsize=16, fontweight='bold')

# ── Plot 1: Confusion Matrix ─────────────────
ax1 = axes[0, 0]
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=present_names, yticklabels=present_names,
            ax=ax1, linewidths=0.3, cbar_kws={'shrink': 0.8})
ax1.set_title('Confusion Matrix (Normalized)', fontweight='bold')
ax1.set_xlabel('Predicted Label')
ax1.set_ylabel('True Label')
ax1.tick_params(axis='both', labelsize=7)

# ── Plot 2: Per-class Accuracy Bar ───────────
ax2 = axes[0, 1]
colors = ['#e74c3c' if a < 0.90 else '#2ecc71' for a in per_class_acc]
bars = ax2.bar(present_names, per_class_acc * 100, color=colors, edgecolor='white', linewidth=0.5)
ax2.axhline(y=90, color='orange', linestyle='--', linewidth=1.2, label='90% threshold')
ax2.axhline(y=acc*100, color='blue', linestyle='--', linewidth=1.2, label=f'Overall {acc*100:.1f}%')
ax2.set_title('Per-Class Accuracy', fontweight='bold')
ax2.set_xlabel('Class')
ax2.set_ylabel('Accuracy (%)')
ax2.set_ylim(0, 105)
ax2.legend(fontsize=8)
ax2.tick_params(axis='x', labelsize=7, rotation=45)

# ── Plot 3: Mean Confidence per class ────────
ax3 = axes[1, 0]
conf_labels = list(per_class_conf.keys())
conf_means  = [per_class_conf[l]['mean_conf'] for l in conf_labels]
conf_colors = ['#e74c3c' if c < 0.80 else '#3498db' for c in conf_means]
ax3.bar(conf_labels, [c*100 for c in conf_means], color=conf_colors, edgecolor='white')
ax3.axhline(y=80, color='orange', linestyle='--', linewidth=1.2, label='80% threshold')
ax3.set_title('Mean Prediction Confidence per Class', fontweight='bold')
ax3.set_xlabel('Class')
ax3.set_ylabel('Mean Confidence (%)')
ax3.set_ylim(0, 105)
ax3.legend(fontsize=8)
ax3.tick_params(axis='x', labelsize=7, rotation=45)

# ── Plot 4: Sample count per class ───────────
ax4 = axes[1, 1]
sample_labels = list(class_counts.keys())
sample_counts = list(class_counts.values())
sc_colors = ['#e74c3c' if s < 500 else '#2ecc71' for s in sample_counts]
ax4.bar(sample_labels, sample_counts, color=sc_colors, edgecolor='white')
ax4.axhline(y=500, color='orange', linestyle='--', linewidth=1.2, label='500 min recommended')
ax4.set_title('Dataset Sample Count per Class', fontweight='bold')
ax4.set_xlabel('Class')
ax4.set_ylabel('Number of Images')
ax4.legend(fontsize=8)
ax4.tick_params(axis='x', labelsize=7, rotation=45)

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=150, bbox_inches='tight')
print("  Saved → model_evaluation.png")
plt.show()

# ─────────────────────────────────────────────
#  10. FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
print(f"  Overall Accuracy      : {acc*100:.2f}%")
print(f"  Top-3 Accuracy        : {top3*100:.2f}%")
print(f"  Top-5 Accuracy        : {top5*100:.2f}%")
print(f"  Total Parameters      : {total_params:,}")
print(f"  Total Images Tested   : {len(X)}")
print(f"  Classes with <90% acc : {sum(1 for a in per_class_acc if a < 0.90)}")
print(f"  Classes with <80% conf: {sum(1 for c in conf_means if c < 0.80)}")
print("\n  ⚠  Classes marked with ⚠ in confidence table")
print("     need more training data or background variety.")
print("=" * 60)