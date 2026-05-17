# Part 2 — Computer Vision: CNN-Based Manufacturing Defect Classifier

---

## Problem Statement
Build a Convolutional Neural Network (CNN) to classify product surface images into one of four defect categories, demonstrating how CNNs learn visual patterns through convolution, pooling, activation, and dense layers.

---

## Dataset
| Property | Value |
|----------|-------|
| File | `labels.csv` + `images/` folder |
| Total images | 480 |
| Classes | 4 — `normal`, `scratch`, `dent`, `stain` |
| Images per class | 120 (perfectly balanced) |
| Image size | 64 × 64 × 3 (RGB) |
| Task type | Multi-class Image Classification |

---

## CNN Architecture
```
Input: 64×64×3
  │
  ├─ Conv2D(32, 3×3) + ReLU  →  MaxPooling2D(2×2)
  ├─ Conv2D(64, 3×3) + ReLU  →  MaxPooling2D(2×2)
  ├─ Conv2D(128,3×3) + ReLU  →  MaxPooling2D(2×2)
  │
  ├─ Flatten
  ├─ Dense(256) + ReLU + Dropout(0.5)
  ├─ Dense(128) + ReLU
  └─ Dense(4)  + Softmax
```

| Component | Choice | Reason |
|-----------|--------|--------|
| Activation | ReLU | Avoids vanishing gradient; sparse activations |
| Pooling | MaxPooling2D | Reduces spatial size; translation invariance |
| Loss | Categorical Cross-Entropy | Standard for multi-class |
| Optimizer | Adam (lr=0.001) | Adaptive, robust default |
| Output | Softmax | Probability distribution over 4 classes |

---

## Results
| Metric | Value |
|--------|-------|
| Final Train Accuracy | ~93% |
| Final Val Accuracy | ~87% |
| Test Accuracy | ~87% |

---

## Repository Structure
```
part-2-cnn-computer-vision/
├── README.md
├── notebook.ipynb
├── requirements.txt
├── sample_predictions/
│   ├── dataset_samples.png
│   └── prediction_outputs.png
└── results/
    ├── accuracy_loss_curves.png
    └── confusion_matrix.png
```

---

## How to Run
```bash
git clone <your-repo-url>
cd part-2-cnn-computer-vision
pip install -r requirements.txt

# Place your dataset:
# labels.csv in root
# images/normal/, images/scratch/, images/dent/, images/stain/ folders

jupyter notebook notebook.ipynb
```

---

## CNN Concept Explanations

### What is Convolution?
A **filter** (small matrix, e.g. 3×3) slides across the image computing dot products at each position, producing a **feature map** that highlights where specific patterns (edges, curves, textures) appear. The CNN learns the best filter values during training — no hand-crafting needed.

### Why is Pooling Used?
**MaxPooling** shrinks feature maps by keeping only the strongest activation in each region (e.g. 2×2 → 1 value). This reduces computation, combats overfitting, and makes the model **spatially invariant** — a scratch in the top-left or bottom-right of an image will still be detected.

### Why is ReLU Commonly Used in CNNs?
- `f(x) = max(0, x)` — computationally trivial
- Does **not saturate** for positive inputs → gradients flow freely → deep networks train efficiently
- Creates **sparse activations** — only neurons that detect the feature fire
- Sigmoid/Tanh suffer from vanishing gradients in deep networks; ReLU does not

### Why are CNNs Better than Feed-Forward Networks for Images?
| Property | Feed-Forward NN | CNN |
|----------|----------------|-----|
| Preserves spatial structure | ❌ (flattens to 1D) | ✅ (2D convolution) |
| Parameter count (64×64 input) | ~3M for first layer | ~864 (32 filters × 3×3) |
| Translation invariance | ❌ | ✅ (pooling + weight sharing) |
| Hierarchical feature learning | ❌ | ✅ (edges→shapes→objects) |

---

## Business Use Case: Manufacturing Quality Inspection

**Problem:** Manual surface inspection is slow (2–3 sec/part), inconsistent (fatigue), and unscalable.

**Solution:** Deploy the CNN on a camera above the conveyor belt. Each product is photographed and classified in real-time (<0.1 sec). Defective units are automatically rejected.

| Metric | Before AI | After AI |
|--------|-----------|----------|
| Inspection speed | 2–3 sec/part | <0.1 sec/part |
| Defect escape rate | 3–5% | <0.5% |
| Labour cost | 8 inspectors/shift | 1 supervisor |
| Consistency | Variable | 100% consistent |

**Responsible AI:** Retrain quarterly as product lines evolve; route unknown defect types to human review; ensure training data covers all lighting conditions and batches.
