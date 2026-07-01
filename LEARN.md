# 🧠 LEARN.md — A Complete, Plain-English Walkthrough of the Brain Tumor Classifier

> A deep-dive study guide for understanding **every part** of this project — the
> concepts, the code, and the "why" behind each decision. Written for the
> EE7204/EC7205 evaluation. Read top-to-bottom, or jump via the contents.

## Contents
1. [The 30-second summary](#1-the-30-second-summary)
2. [The problem & why deep learning](#2-the-problem--why-deep-learning)
3. [The data](#3-the-data)
4. [Preprocessing — `src/preprocess.py`](#4-preprocessing--srcpreprocesspy)
5. [Augmentation — `src/augment.py`](#5-augmentation--srcaugmentpy)
6. [The models — `src/models.py`](#6-the-models--srcmodelspy)
7. [Training — `src/train.py`](#7-training--srctrainpy)
8. [Evaluation — `src/evaluate.py`](#8-evaluation--srcevaluatepy)
9. [The results, explained](#9-the-results-explained)
10. [The plots (matplotlib)](#10-the-plots-matplotlib)
11. [The UIs](#11-the-uis)
12. [How to run everything](#12-how-to-run-everything)
13. [Viva / exam Q&A](#13-viva--exam-qa)

---

## 1. The 30-second summary

You take a **brain MRI image** and answer one question: *which of 4 categories is
this?* → **glioma**, **meningioma**, **pituitary tumor**, or **no tumor**.

The system does it in 5 stages:

```
Raw MRI ──▶ Preprocess ──▶ Augment ──▶ Train 4 CNNs ──▶ Evaluate & Compare ──▶ UI demo
 (image)    (clean it)    (train only)  (learn)         (measure accuracy)     (predict)
```

Four models are trained and compared:

| Model | What it is | Test accuracy |
|-------|-----------|---------------|
| **ResNet50** ⭐ | Pre-trained, fine-tuned (winner) | **85.8%** |
| VGG16 | Pre-trained, fine-tuned | 84.6% |
| EfficientNetB0 | Pre-trained, fine-tuned | 83.8% |
| Custom CNN | Built & trained from scratch | 71.7% |

**Headline lesson:** transfer learning (reusing a network already trained on
1.2M ImageNet photos) beats training from scratch when you only have ~1,600
images. That's the whole point of comparing them.

---

## 2. The problem & why deep learning

- **Medical need:** brain tumors are life-threatening; early detection matters.
  Radiologists reading MRIs manually are slow and can disagree — especially for
  glioma vs meningioma, which *look similar*.
- **Goal:** an automated "second opinion" that classifies a scan in <1 second.
- **Why a CNN (Convolutional Neural Network)?** Classic image processing needs a
  human to hand-design features ("look for bright blobs of size X"). A CNN
  **learns** the useful features by itself from labelled examples. For subtle
  texture differences between tumor types, learned features win.

**Key term — classification vs segmentation:** this project does
*classification* (one label per whole image), **not** *segmentation* (drawing the
exact tumor outline pixel-by-pixel). Segmentation is listed as future work.

---

## 3. The data

- **Source:** Kaggle "Brain Tumor MRI Dataset" (~7,023 images).
- **Shipped in this repo:** a balanced **1,600-image** subset → 400 per class.
- **Split (stratified 70/15/15):** 1,120 train / 240 val / 240 test
  (280 / 60 / 60 per class).

**Three key vocabulary words** — you *will* be asked this:

| Set | Size | Job | Analogy |
|-----|------|-----|---------|
| **Train** | 1,120 | Model learns from these | Homework you study |
| **Validation** | 240 | Tune settings, decide when to stop | Practice exam |
| **Test** | 240 | Final unbiased score — touched once | The real exam |

> **Stratified** split = each set keeps the same class proportions (25% each), so
> no set accidentally gets too few gliomas. Done with a fixed `random_state=42`
> so the split is **reproducible** (same every run).

```python
# src/preprocess.py — the two-step stratified split
X_train, X_temp, y_train, y_temp = train_test_split(
    paths, labels, train_size=0.70, stratify=labels, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)  # 15% / 15%
```

---

## 4. Preprocessing — `src/preprocess.py`

This is the **image-processing heart** of the project (the part your IP&CV
course cares most about). Every raw MRI goes through 7 steps before a model ever
sees it. The goal: **remove everything that isn't useful signal**, so the model
learns tumor patterns, not scanner quirks.

The full pipeline in one function:

```python
def preprocess_image(image_path, return_stages=False):
    raw     = load_image(image_path)            # 1. read from disk (BGR uint8)
    gray3   = to_grayscale_rgb(raw)             # 2. grayscale, copied into 3 channels
    resized = resize_image(gray3, 224)          # 3. force 224x224
    gauss   = apply_gaussian_filter(resized, 3) # 4. smooth random noise
    median  = apply_median_filter(gauss, 3)     # 5. kill salt-and-pepper noise, keep edges
    stripped= skull_strip(median)               # 6. cut away the skull
    norm    = normalize_image(stripped)         # 7. scale pixels to [0,1]
    return {...} if return_stages else norm
```

### Step-by-step (the "why" matters more than the "how")

**1. Load** — `cv2.imread` reads the image as a NumPy array. OpenCV loads in
**BGR** order (not RGB) — remember this, it's a classic gotcha when displaying.

**2. Grayscale → 3 channels** — MRIs are grayscale (1 channel), but the
pre-trained models expect a 3-channel (colour) input because they were trained on
colour photos. So we copy the single gray channel three times:
```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rgb  = cv2.merge([gray, gray, gray])   # same value in R, G, B
```

**3. Resize to 224×224** — VGG16/ResNet50/EfficientNet were all designed for
224×224 inputs, so every image must match. Uses `INTER_AREA` (best for shrinking).

**4. Gaussian filter (3×3)** — blurs *slightly* to smooth high-frequency random
("Gaussian") noise from the scanner. Each pixel becomes a weighted average of its
neighbours: `cv2.GaussianBlur(img, (3,3), 0)`.

**5. Median filter (3×3)** — replaces each pixel with the **median** of its
neighbourhood. This is the classic cure for **salt-and-pepper** (impulse) noise,
and crucially it **preserves edges** (tumor boundaries) far better than blurring.
```python
cv2.medianBlur(img, 3)
```
> **Exam favourite — Gaussian vs Median:** Gaussian = weighted *average*, great
> for Gaussian noise, but *smears* edges. Median = middle value, great for impulse
> noise, and *keeps* edges sharp. Using both = belt and braces.

**6. Skull stripping** — the skull/scalp carries no tumor information and can
distract the model. Algorithm (all classic OpenCV):
```python
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # auto threshold
closed    = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)  # fill holes
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest   = max(contours, key=cv2.contourArea)   # the brain is the biggest blob
mask      = draw filled largest contour
stripped  = cv2.bitwise_and(img, mask)           # keep only inside the brain
# then crop to the brain's bounding box, pad to square, resize back to 224
```
> **Otsu's threshold** automatically picks the brightness cut-off that best
> separates "bright brain" from "dark background" — no magic number needed.
> **Morphological closing** = dilate then erode; it fills small black holes so the
> brain becomes one solid blob. **Contours** = outlines of connected shapes; the
> biggest one is the brain.

**7. Normalize (Min-Max to [0,1])** — neural nets train more stably when inputs
are small and consistent. Map `[0,255] → [0,1]`:
```python
img_f = (img_f - img_f.min()) / (img_f.max() - img_f.min())
```

Processed images are saved as PNG into `data/processed/{train,val,test}/{class}/`.

---

## 5. Augmentation — `src/augment.py`

**Problem:** ~1,120 training images is *tiny* for deep learning → the model
memorises them (**overfitting**) instead of learning general patterns.

**Fix:** on each epoch, randomly distort training images so the model sees fresh
variations and can't memorise. Applied **only to training data**, never to
val/test (those must stay pristine for honest scoring).

```python
augmentation = keras.Sequential([
    keras.layers.RandomRotation(0.042),      # ±15°  (head tilt)
    keras.layers.RandomFlip("horizontal"),   # left/right mirror
    keras.layers.RandomZoom(0.12),           # ±12%  (tumor size varies)
    keras.layers.RandomTranslation(0.10, 0.10), # ±10% shift (not always centred)
    keras.layers.RandomBrightness(0.20),     # ±20%  (scanner differences)
    keras.layers.RandomContrast(0.20),       # contrast wobble
])
```

> **Why the values are *deliberately mild*:** the code comments explain that
> **aggressive** augmentation *hurt* glioma/meningioma accuracy — big rotations
> and brightness swings wash out the subtle texture differences that separate
> those two look-alike classes. So rotation was tuned down ±20°→±15°, brightness
> ±30°→±20°, etc. Good talking point: *"we tuned augmentation strength to the
> data."*

---

## 6. The models — `src/models.py`

Four architectures, one shared idea: **stack of feature extractors → classifier
head → 4-way softmax**.

### CNN building blocks (know these cold)

- **Conv2D** — slides small learnable filters over the image to detect patterns
  (edges → textures → shapes as you go deeper).
- **ReLU** — activation `max(0, x)`; adds non-linearity so the net can learn
  complex functions.
- **BatchNormalization** — re-centres each layer's outputs; speeds up and
  stabilises training.
- **MaxPooling2D** — downsamples (keeps the max in each 2×2 block); shrinks the
  image and adds small translation-invariance.
- **Dropout** — randomly switches off a fraction of neurons during training so the
  net can't over-rely on any one → fights overfitting.
- **GlobalAveragePooling / Flatten** — turn the 2D feature maps into a 1D vector
  for the final dense layers.
- **Softmax** — final layer turns 4 raw scores into 4 probabilities that sum to
  100%.

### Model 1 — Custom CNN (built from scratch)

Five conv blocks with growing filters (32→64→128→256→512), then a 1×1 conv to
shrink channels, then Flatten → Dense → Dropout → softmax.

```python
x = layers.Conv2D(32,(3,3),padding="same")(inputs)   # block 1
x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
x = layers.MaxPooling2D((2,2))(x)
# ... blocks 2–5 double the filters each time ...
x = layers.Conv2D(64,(1,1))(x)     # 1x1 conv: squeeze 512→64 channels (fewer params)
x = layers.Flatten()(x)            # keep spatial layout (helps locate pituitary/meningioma)
x = layers.Dense(512, "relu")(x); x = layers.Dropout(0.5)(x)
x = layers.Dense(256, "relu")(x); x = layers.Dropout(0.5)(x)
outputs = layers.Dense(4, "softmax")(x)
```
> **Design note in the code:** they chose **Flatten over GlobalAveragePooling**
> because GAP throws away *where* things are, and location helps (pituitary sits
> centrally, meningioma at the periphery). The 1×1 conv first prevents a parameter
> explosion (7×7×512 = 25k features → 7×7×64 = 3,136).

### Models 2–4 — Transfer Learning (VGG16, ResNet50, EfficientNetB0)

**The big idea:** don't start from zero. Take a network already trained on
**ImageNet** (1.2M everyday photos) — it already knows edges, textures, shapes —
and *reuse* it, retraining only a small new "head" for our 4 classes.

```python
base = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
for layer in base.layers:          # freeze the borrowed knowledge
    layer.trainable = False
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512,"relu", kernel_regularizer=l2(1e-4))(x)  # our new head
x = layers.BatchNormalization()(x); x = layers.Dropout(0.4)(x)
x = layers.Dense(256,"relu", kernel_regularizer=l2(1e-4))(x)
x = layers.BatchNormalization()(x); x = layers.Dropout(0.3)(x)
outputs = layers.Dense(4,"softmax")(x)
```

**Two-stage fine-tuning** (the clever bit):
- **Stage 1** — freeze the whole ImageNet backbone, train *only* the new head at a
  normal learning rate (1e-3). Fast, and it gets the head roughly right.
- **Stage 2** — unfreeze the *top few* backbone layers and train everything at a
  *tiny* learning rate (1e-5) so the borrowed features gently adapt to MRIs
  without being destroyed. Unfreeze depth differs per model: VGG16 last 8,
  ResNet50 last 33, EfficientNet last 50.

> **Why a tiny LR in Stage 2?** Big updates would wreck the carefully-learned
> ImageNet weights ("catastrophic forgetting"). Small nudges = safe adaptation.
> **Why keep BatchNorm layers frozen?** To preserve the ImageNet mean/variance
> statistics; letting them shift on a small dataset destabilises training.
> The three architectures differ: **VGG16** = simple deep stack; **ResNet50** =
> "residual" skip-connections let it go 50 layers deep without vanishing
> gradients; **EfficientNet** = mathematically balanced depth/width/resolution →
> best accuracy-per-parameter.

---

## 7. Training — `src/train.py`

### The input pipeline (`tf.data`)

```python
ds = keras.utils.image_dataset_from_directory(split_dir, label_mode="categorical",
        class_names=CLASS_NAMES, image_size=(224,224), batch_size=32)
ds = ds.map(lambda x,y: (x/255.0, y))                  # 1. to [0,1]
if augment: ds = ds.map(lambda x,y: (aug_model(x, training=True), y))  # 2. augment (train only)
ds = ds.map(lambda x,y: (tf.clip_by_value(x,0,1), y))  # 3. clip brightness overshoot
if transfer: ds = ds.map(lambda x,y: (preprocess_fn(x*255), y))  # 4. backbone-specific prep
```
Note step 4: each backbone wants inputs prepared *its* way (e.g. EfficientNet
expects [-1,1]), so we scale back to [0,255] and call that model's
`preprocess_input`. The Custom CNN just uses [0,1] directly.

### Training settings (and the reasons)

- **Optimizer: Adam** — the reliable default; adapts the step size per weight.
- **Loss: Categorical Cross-Entropy with `label_smoothing=0.1`** — the standard
  loss for multi-class. Label smoothing means the target is 0.9 instead of 1.0 →
  discourages the model from being *overconfident*, improves generalisation.
- **Class weights (balanced)** — even though the subset is balanced, this
  up-weights harder/under-represented classes so they aren't ignored.
- **Callbacks:**
  - `EarlyStopping(patience=8, restore_best_weights=True)` — stop when validation
    loss stops improving and **roll back** to the best epoch (so extra epochs
    can't hurt).
  - `ModelCheckpoint` — save the best model to disk.
  - `ReduceLROnPlateau(factor=0.5, patience=3)` — when val-loss stalls, halve the
    learning rate to fine-tune.
  - `TensorBoard` — logs curves (that's what's in `logs/`).

> **Bug-fix lore (great to mention — shows engineering maturity):** the comments
> record real fixes: (BUG-1) CosineDecay LR schedule crashed `ReduceLROnPlateau`,
> so Stage 2 uses a plain float LR; (BUG-2) `RandomBrightness` could push pixels
> above 1.0, so they clip; (BUG-3) Stage 2 loads the *full* saved Stage-1 model
> instead of just weights to avoid silent mismatches.

The custom CNN trains in a **single stage** (80 epochs); transfer models train in
**two stages** (30 + 30). Every model saves `models/{name}_final.keras` and a
`{name}_history.json` for the curves.

---

## 8. Evaluation — `src/evaluate.py`

Runs each trained model on the **untouched test set** and computes the four
metrics your proposal requires. Predictions come from `argmax` over the softmax.

### The four metrics — with a concrete intuition

Imagine the model's job is "find gliomas":

| Metric | Formula | Plain meaning |
|--------|---------|---------------|
| **Accuracy** | correct / total | Overall, how often is it right? |
| **Precision** | TP / (TP+FP) | *When it says "glioma", how often is it correct?* |
| **Recall** | TP / (TP+FN) | *Of all real gliomas, how many did it catch?* |
| **F1-score** | 2·P·R/(P+R) | Balance of precision & recall (one number) |

(TP=true positive, FP=false positive, FN=false negative.) In medicine **recall
matters** — missing a real tumor (low recall) is dangerous.

> **"Weighted average"** in the results = average the per-class metric weighted by
> class size. With a balanced test set it's just the plain average.

### Confusion matrix

A 4×4 grid: rows = true class, columns = predicted class. The **diagonal** =
correct; **off-diagonal** = mistakes. In this project the glioma↔meningioma cell
is always the hottest off-diagonal — the model's main confusion.

```python
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
```

---

## 9. The results, explained

From `reports/model_comparison.csv` (test set, 240 images):

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| **ResNet50** ⭐ | **85.8%** | 86.5% | 85.8% | 85.7% |
| VGG16 | 84.6% | 86.1% | 84.6% | 84.5% |
| EfficientNetB0 | 83.8% | 84.5% | 83.8% | 83.3% |
| Custom CNN | 71.7% | 70.0% | 71.7% | 69.8% |

**Three things to say about these numbers:**

1. **Transfer learning >> from scratch** (85.8% vs 71.7%). With only ~1,120
   training images, borrowing ImageNet features wins decisively. This is the
   central finding.
2. **`notumor` and `pituitary` are easy** (F1 ≈ 0.87–0.96 everywhere).
   **`glioma` vs `meningioma` is hard** (they share texture) — that's where every
   model loses points, and it matches the medical literature.
3. **Custom CNN's meningioma recall collapses to ~30%** (F1 0.38) — a small
   from-scratch net just can't separate the two look-alike tumors. See it plainly
   in `reports/per_class_f1_heatmap.png`.

> **Honesty note:** these are on the **1,600-image subset**. On the full ~7,000
> image Kaggle set the same pipeline is expected to exceed 90% — worth stating so
> the numbers aren't undersold.

---

## 10. The plots (matplotlib)

Two scripts produce the charts, both pure/mostly matplotlib:

- **`src/evaluate.py`** — needs the trained models; produces confusion matrices
  and training curves (per-model) plus the comparison chart.
- **`generate_metrics_plots.py`** *(added for this project)* — needs **no models**;
  rebuilds every metric chart instantly from the saved
  `reports/classification_reports.json` + `reports/model_comparison.csv`:
  - `overall_metrics_comparison.png` — acc/precision/recall/F1 per model
  - `accuracy_leaderboard.png` — ranked horizontal bars
  - `per_class_f1_comparison.png` — F1 per class, all models
  - `per_class_metrics_resnet50.png` — P/R/F1 for the best model
  - `per_class_f1_heatmap.png` — model×class heatmap (great one-glance summary)

```bash
python generate_metrics_plots.py     # regenerate all metric plots in ~1 second
```

Because it reads only the saved CSV/JSON, you can regenerate these for your slides
even on a laptop with no GPU and no trained models present.

---

## 11. The UIs

Two Streamlit apps:

- **`src/app.py`** — the full-featured UI: dark theme, plotly charts, clinical
  info cards, per-class table.
- **`src/app_minimal.py`** *(added for this project)* — a small, easy-to-explain
  demo. Upload → 6-stage preprocessing gallery → prediction + a matplotlib
  probability bar chart. Importantly it **still runs the preprocessing demo even
  with no trained model**, and only enables prediction once a model exists.

```bash
streamlit run src/app_minimal.py     # minimal demo (recommended for the viva)
streamlit run src/app.py             # full UI
```

Both apply the *same* preprocessing and the *same* backbone-specific input prep as
training — that consistency is what makes predictions valid.

---

## 12. How to run everything

```bash
# 0. activate the environment
source venv/bin/activate

# 1. (data already shipped) preprocess raw → processed  [optional, already done]
python -m src.preprocess

# 2. train models (GPU strongly recommended; CPU is slow)
python -m src.train                       # all four
python -m src.train --models custom_cnn   # just one (fastest)

# 3. evaluate → writes reports/ (needs trained models)
python -m src.evaluate

# 4. regenerate metric plots from saved reports (no models needed)
python generate_metrics_plots.py

# 5. demo UI
streamlit run src/app_minimal.py
```

> **This clone has no trained models** (`models/` is git-ignored). Preprocessing,
> the metric plots, and the preprocessing half of the UI all work now; **live
> prediction needs you to run step 2 first.** Training the Custom CNN is the
> fastest way to get one working model for the demo.

---

## 13. Viva / exam Q&A

**Q: Why resize to 224×224?** Because VGG16/ResNet50/EfficientNet were designed
for that size; the input must match the network.

**Q: Why replicate grayscale into 3 channels?** The pre-trained backbones expect
3-channel input (trained on colour photos); copying the gray channel 3× satisfies
that without inventing colour.

**Q: Gaussian vs median filter — when/why?** Gaussian = weighted average, removes
Gaussian noise but blurs edges. Median = middle value, removes salt-and-pepper
noise and preserves edges (tumor boundaries). We use both.

**Q: What is Otsu's method?** An automatic thresholding technique that chooses the
intensity cut-off minimising within-class variance — separates brain from
background without a hand-picked number.

**Q: What is transfer learning and why does it help here?** Reusing a network
pre-trained on a huge dataset (ImageNet) and adapting it to our task. It helps
because our dataset is small; the borrowed low-level features (edges, textures)
transfer well and we avoid overfitting.

**Q: Why two-stage training?** Stage 1 trains the new head safely with the
backbone frozen; Stage 2 gently fine-tunes the top backbone layers at a tiny
learning rate so pre-trained knowledge adapts without being destroyed.

**Q: What stops overfitting here?** Data augmentation, Dropout, L2 regularisation,
BatchNorm, EarlyStopping with best-weight restore, and class weighting.

**Q: Accuracy vs precision vs recall vs F1?** See the table in §8. In medicine,
recall (catching real tumors) is critical.

**Q: Why is glioma vs meningioma the hardest pair?** Their MRI textures overlap;
the confusion matrix and the F1 heatmap both show this is where errors cluster.

**Q: Why did ResNet50 win?** Residual (skip) connections let it train deep without
vanishing gradients, extracting richer features than VGG16 or the from-scratch CNN
— matching the proposal's expectation.

**Q: What would improve results?** Use the full 7k-image dataset, add Grad-CAM for
explainability, try an ensemble of the models, or move to segmentation (U-Net).
```
