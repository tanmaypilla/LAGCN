# LAGCN: Architecture, CPR, and Training — Technical Notes

---

## Table of Contents

1. [Skeleton Training Pipeline](#1-skeleton-training-pipeline)
2. [Data Preprocessing — The Feeder](#2-data-preprocessing--the-feeder)
3. [Model Architecture](#3-model-architecture)
4. [CPR: What It Is and How It's Built from Text](#4-cpr-what-it-is-and-how-its-built-from-text)
5. [The Auxiliary Branch — How CPR Enters Training](#5-the-auxiliary-branch--how-cpr-enters-training)
6. [The Einsum Operation — Deep Dive](#6-the-einsum-operation--deep-dive)
7. [How CPR Shapes the Gradients](#7-how-cpr-shapes-the-gradients)
8. [Diagnostic Results](#8-diagnostic-results)
9. [Why FACEOFF Was Hard in the First Place](#9-why-faceoff-was-hard-in-the-first-place)
10. [Known Implementation Issues](#10-known-implementation-issues)
11. [BERT vs CLIP for CPR Generation](#11-bert-vs-clip-for-cpr-generation)
12. [Experiment Results](#12-experiment-results)
    - [12.1 Temporal CPR vs Baseline](#121-temporal-cpr-vs-baseline)
    - [12.2 Velocity Stream vs Baseline](#122-velocity-stream-vs-baseline)
    - [12.3 Why Joint + Motion Fusion Is the Natural Next Step](#123-why-joint--motion-fusion-is-the-natural-next-step)
    - [12.4 Joint + Motion Fusion Results](#124-joint--motion-fusion-results)
    - [12.5 Data Split Fix](#125-data-split-fix-combined_annotations_v2)
    - [12.6 Fusion + T=30 Window (no accel)](#126-fusion--t30-window-correct-split-no-accel)
    - [12.7 Fusion + T=30 + Acceleration Channel](#127-fusion--t30--acceleration-channel)
    - [12.8 Imbalance Strategies + Sequence Modelling Postprocessing](#128-imbalance-strategies--sequence-modelling-postprocessing)

---

## 1. Skeleton Training Pipeline

### Input Format

Raw skeleton data enters the model as a tensor of shape `(N, C, T, V, M)`:

- **N** = batch size
- **C** = coordinate channels (2 for xy in hockey, 3 for xyz in NTU)
- **T** = temporal frames (resized to 64 for hockey)
- **V** = joints (20 for hockey, 25 for NTU)
- **M** = number of people (1 for hockey, 2 for NTU)

### End-to-End Data Flow

```
Raw skeleton (T frames, V joints, C coords)
    ↓ [Feeder: normalization + temporal crop/resize]
(C, T, V, M) tensor
    ↓ [Joint embedding: Linear(C → 64) + positional encoding]
(N*M*T, V, 64)
    ↓ [Data BN + rearrange]
(N*M, 64, T, V)
    ↓ [9× TCN-GCN layers]
(N*M, 256, T', V)       T' = 16 after two stride-2 layers
    ↓
    ├── MAIN BRANCH: global avg pool → FC(256 → num_classes) → main logits
    └── AUX BRANCH:  einsum with CPR → conv → pool → aux logits
    ↓
Total Loss = CE(main_logits, label) + 0.2 × CE(aux_logits, label)
    ↓ [Backward — gradients flow through both branches into all 9 GCN layers]
```

---

## 2. Data Preprocessing — The Feeder

The hockey feeder (`feeders/feeder_hockey.py`) applies three preprocessing steps before the skeleton reaches the model.

### Step 1: Hip-center subtraction

```python
hip = data[:, :, [5, 6], :].mean(axis=(1, 2, 3), keepdims=False)
data = data - hip
```

Joints 5 and 6 are the left and right hips. Their mean is subtracted from all joints across all frames, centering the skeleton at the hips.

**Why:** Raw keypoints are in camera/image pixel coordinates — absolute positions on the ice. Two players doing identical actions on opposite ends of the rink would have completely different raw coordinate values. The model would waste capacity learning position-invariance rather than motion patterns. After subtraction, every skeleton is anchored at `(0, 0)` regardless of where on the ice the player is.

**Q: Doesn't this muddy the coordinates?**

No — it clarifies them. The GCN operates on edges between joints (joint `i` relative to joint `j`), so the absolute rink position was always noise from its perspective. What matters is body configuration and motion. The only information lost is global rink position, which is irrelevant to action classification. Two identical skating motions should not be classified differently because one player happened to be near the boards.

### Step 2: Max-abs normalization

```python
scale = np.abs(data).max()
data = data / scale
```

After centering, coordinate magnitudes still depend on camera distance — a player close to the camera has larger pixel displacements than one far away, even doing the same movement. Dividing by the max absolute value maps all coordinates to `[-1, 1]`, making motion scale comparable across all samples. It also stabilizes training by keeping features in a range compatible with weight initialization and activation functions.

### Step 3: Temporal crop + bilinear resize

```python
# Train: random crop
p = np.random.uniform(p_interval[0], p_interval[1])
cropped_length = int(p * valid_frame_num)
bias = np.random.randint(0, valid_frame_num - cropped_length + 1)
data = data[:, bias:bias+cropped_length, :, :]

# Resize to window_size=64 via bilinear interpolation
```

Two distinct purposes:

- **Variable-length handling:** Hockey clips vary from 15 to 30 frames. The model requires a fixed-size input tensor, so all clips are resampled to exactly 64 frames via bilinear interpolation (temporal resampling).
- **Data augmentation (train only):** Randomly cropping a fraction `p ∈ [p_min, p_max]` of the clip starting from a random offset forces the model to recognize actions from partial views, prevents over-reliance on clip boundaries, and multiplies the effective dataset size. At test time a deterministic center crop is used instead.

---

## 3. Model Architecture

### Layer Stack

The model uses 9 stacked `TCN_GCN_unit` blocks:

```
l1: TCN_GCN(64→64,   stride=1, loop_times=4)   ← multi-hop aggregation
l2: TCN_GCN(64→64,   stride=1)
l3: TCN_GCN(64→64,   stride=1)
l4: TCN_GCN(64→128,  stride=2)                  ← temporal downsampling: T=64→32
l5: TCN_GCN(128→128, stride=1)
l6: TCN_GCN(128→128, stride=1)
l7: TCN_GCN(128→256, stride=2)                  ← temporal downsampling: T=32→16
l8: TCN_GCN(256→256, stride=1)
l9: TCN_GCN(256→256, stride=1)
```

### TCN_GCN_unit

Each block applies spatial message passing then temporal convolution:

```
GCN (unit_gcn) → TCN (MultiScale_TemporalConv) → ReLU + residual
```

**unit_gcn** processes three adjacency matrix subsets independently and sums their outputs:
- `A[0]`: Self-loops (each joint attends to itself)
- `A[1]`: Inward edges (parent→child in kinematic chain, e.g. hip→knee), degree-normalized
- `A[2]`: Outward edges (child→parent, e.g. wrist→elbow), degree-normalized

**CTRGC** (the core graph convolution inside each subset):
- Computes a dynamic, input-dependent adjacency: `tanh(x_i - x_j)` for all joint pairs
- Combines with static adjacency: `A_effective = A_dynamic * alpha + A_static`
- `alpha` is a learnable scalar, initialized near zero so static graph dominates early training
- Performs: `output[n,c,t,u] = Σ_v A[n,c,u,v] * x[n,c,t,v]`
- l1 additionally applies k-hop aggregation (`loop_times=4`), fusing multi-hop neighbors with exponentially decaying weights

**MultiScale_TemporalConv**: 4 dilated temporal convolutions (dilations 1,2,3,4) + 1 max-pool + 1 identity, all concatenated. Captures motion patterns at multiple timescales simultaneously.

### Skeleton Graph (Hockey)

The hockey skeleton has 20 joints connected by the body's kinematic chain. Edges are predefined (anatomical), stored as inward/outward pairs, then normalized by in-degree. The graph adjacency `A` has shape `(3, 20, 20)`.

---

## 4. CPR: What It Is and How It's Built from Text

### What CPR Is

CPR (Class Prior Relationship) is a **pre-computed, frozen matrix per action class** of shape `(V, V)` — a joint-to-joint similarity matrix that encodes how semantically related each pair of joints is *in the context of that class*, according to a language model.

For hockey: shape `(11, 20, 20)`. Entry `CPR[c, i, j]` answers: *"how semantically related are joint i and joint j when a player is performing class c?"*

CPR is computed once before training, stored as `.npy`, and never updated. It enters training only through the auxiliary branch loss.

### Generation Pipeline

**Step 1 — Template sentence construction**

A template with `[C]` (class) and `[J]` (joint) placeholders is filled in for every `(class, joint)` pair:

```
Template: "when [C] [J] of hockey player."

Examples:
  "when forward gliding left knee of hockey player."
  "when faceoff left wrist of hockey player."
  "when post whistle right shoulder of hockey player."
```

For PASTA-grounded variants, Gemini-generated descriptions of body part behavior are inserted, producing richer sentences like:
```
"faceoff. [Gemini: upper body crouched, stick held vertically at center ice...]. left elbow."
```

This produces `num_classes × num_joints` sentences (11 × 20 = 220 for hockey).

**Step 2 — Text embedding**

Each sentence is encoded by a frozen pretrained language model:
- **BERT** (`bert-base-uncased`): takes the `pooler_output` — the `[CLS]` token's final hidden state after a linear+tanh layer — giving a 512-dim vector per sentence
- **CLIP** (`ViT-B/32`): uses the text encoder, also giving a 512-dim vector

Result: for each class `c`, a set of `V` vectors of shape `(512,)`, one per joint.

**Step 3 — Pairwise cosine similarity**

For each class `c`, stack its V joint embeddings into `(V, 512)`, L2-normalize, then:

```python
mat = (joint_embeddings @ joint_embeddings.T) * 200   # (V, V) cosine similarity × temperature
```

`mat[i, j]` = cosine similarity between the text embedding for joint `i` and joint `j` in the context of class `c`, scaled by temperature 200.

**Step 4 — Save**

Stack all class matrices: `np.save('...npy', np.stack(all_mats))` → shape `(num_classes, V, V)`.

### Visualization Findings (PASTA-grounded BERT)

Running CPR visualization (`visualize_cpr.py`) on the hockey dataset revealed:

- **Skating classes** (GLID_FORW, ACCEL_FORW) have high-contrast CPR matrices with clear joint-pair clusters. The text model encodes rich joint co-activation for these classes.
- **FACEOFF and POST_WHISTLE** have nearly flat CPR matrices — all values close to a uniform constant. The text model produces nearly identical embeddings for all joint descriptions regardless of the faceoff/post-whistle context.
- **Inter-class cosine similarity** is very high (~0.95–1.0) across all classes. Text embeddings barely distinguish classes in embedding space at all.
- **Joint importance**: ear, nose, shoulder joints consistently dominate. Stick joints are uniformly low-importance. The text descriptions are more expressive about upper-body joints.

---

## 5. The Auxiliary Branch — How CPR Enters Training

### The Two Branches

After the 9 GCN layers produce `(N*M, 256, T', V)`, the model splits into two parallel classification paths:

**Main branch** (80% of loss):
```
global avg pool over T and V → (N, 256)
FC(256 → num_classes) → main logits (N, num_classes)
```
Standard classification. Answers: *"what class is this skeleton?"*

**Auxiliary branch** (20% of loss):
```
[time pool / temporal einsum with CPR]
→ (N*M, 256, num_classes, V)
→ Conv2d(256→1) → (N*M, 1, num_classes, V)
→ squeeze + mean over joints → aux logits (N, num_classes)
```
CPR-filtered classification. Answers: *"does this skeleton's joint activation pattern match what the text says class c looks like?"*

### Why the Auxiliary Branch Exists

Without the auxiliary branch, CPR has zero effect on training. The CPR matrices sit in memory and nothing references them. The auxiliary branch is the **only mechanism** by which text knowledge enters the training loop.

The loss is:
```
Total Loss = CE(main_logits, label) + 0.2 × CE(aux_logits, label)
```

The 0.2 weight means the aux branch acts as a regularizer — it doesn't dominate but its gradients consistently bias the GCN layers toward learning joint features that align with the text-derived joint co-activation patterns.

**Q: Would increasing aux_weight help FACEOFF/POST_WHISTLE?**

No. When CPR is flat (all values nearly equal), the einsum output reduces to a scaled global sum: `Σ_v features[v] * constant` — completely uninformative about class identity. The aux logits for flat-CPR classes are near-random. Increasing aux_weight to 0.5 or higher would:
- Push stronger informative gradients for well-conditioned classes (GLID_FORW, ACCEL_FORW) → those improve
- Push stronger noise gradients for flat-CPR classes (FACEOFF, POST_WHISTLE) → those get worse

A uniform weight increase widens the performance gap rather than closing it.

---

## 6. The Einsum Operation — Deep Dive

### Original Aux Branch (Static CPR)

After the 9 GCN layers:
```
x: (N*M, 256, T', V)
→ x.mean(2) → aux_x: (N*M, 256, V)    # collapse time into one static snapshot
→ einsum 'nmv,cvu->nmcu' with CPR (num_classes, V, V)
→ aux_x: (N*M, 256, num_classes, V)
```

In the einsum string, letter assignments are positional (not semantic):
- `n` = N*M (batch dimension)
- `m` = 256 (feature channels)
- `v` = input joint index (contracted over)
- `c` = class index
- `u` = output joint index

The computation for one output element:
```
aux_x[n, m, c, u] = Σ_v   features[n, m, v]  ×  CPR[c, v, u]
```

For class `c`, output joint `u` collects features from all input joints `v`, weighted by `CPR[c, v, u]` — how strongly the text says joint `v` connects to joint `u` for class `c`.

**Concrete example** — FACEOFF (c=7), left elbow (u=7):
```
output[n, m, 7, 7] = features[n, m, R_ear]    × CPR[7, R_ear,    L_elbow]
                   + features[n, m, L_ear]    × CPR[7, L_ear,    L_elbow]
                   + features[n, m, nose]     × CPR[7, nose,     L_elbow]
                   + features[n, m, R_shoulder]× CPR[7, R_shoulder,L_elbow]
                   + ...  (20 terms total)
```

The result is 11 different views of the same skeleton — one per class, each filtered through that class's text-derived joint co-activation pattern. `aux_fc` (a 1×1 conv) then collapses 256 channels to 1 scalar per `(class, joint)`, and mean over joints gives one logit per class.

### Temporal CPR (New)

The static CPR path collapses T before the einsum, discarding all temporal information. The temporal CPR path applies CPR at every timestep before pooling:

```python
# (N*M, C, T', V) × (num_classes, V, V) → (N*M, C, T', num_classes, V) → (N*M, C, num_classes, V)
aux_x = torch.einsum('bdtv,cvu->bdtcu', x, self.examplar).mean(2)
```

Letter assignments:
- `b` = N*M (batch)
- `d` = 256 (channels)
- `t` = T' (time, 16 frames)
- `v` = input joint (contracted)
- `c` = class
- `u` = output joint

Both paths produce `(N*M, 256, num_classes, V)` into `aux_fc` — the downstream layers are identical.

**Why this matters for FACEOFF/POST_WHISTLE:** These are event classes defined by a specific moment in a sequence (the crouch into a faceoff, the drift after a whistle). The mean-pool collapses the discriminative temporal moment into an average that looks like slow gliding. Applying CPR per-timestep lets the model ask *"does the joint pattern at this specific frame match the FACEOFF CPR?"* — brief moments of a distinctive configuration contribute signal even if the rest of the clip looks ambiguous.

**Memory cost:** At T'=16, batch=64, the intermediate tensor `(64, 256, 16, 11, 20)` is ~228MB float32. Significant but manageable on an 8GB+ GPU.

---

## 7. How CPR Shapes the Gradients

Since CPR is frozen, the backward pass through the einsum gives:

```
∂L_aux / ∂features[n, d, v] = Σ_{c,u}  (∂L_aux / ∂aux_x[n, d, c, u])  ×  CPR[c, v, u]
```

The gradient flowing back to joint `v`'s features is a **CPR-weighted sum of classification error signals across all classes**:

- **High CPR[c, v, u]**: joint `v` receives a strong gradient from class `c`'s error signal. When the model misclassifies class `c`, joint `v` is pushed to update toward better class-`c` discrimination.
- **Low CPR[c, v, u]**: joint `v` gets weak gradient from class `c`'s aux loss.

The net effect: CPR acts as a **structural regularizer** that biases which joints should co-activate for each class. GCN layers are nudged to learn joint features consistent with the text model's view of joint co-activation.

Both the main and auxiliary branches backpropagate through all 9 GCN layers simultaneously. The main branch drives classification with full weight; the auxiliary branch overlays a text-prior-shaped gradient at 20% strength.

---

## 8. Diagnostic Results

A feature-space diagnostic (`diagnose_features.py`) was run on three trained models to determine whether FACEOFF/POST_WHISTLE failure is due to flat CPR or fundamentally indistinguishable features.

### What the Diagnostic Measures

1. **t-SNE** of the 256-dim pre-FC features — do FACEOFF/POST_WHISTLE form their own clusters?
2. **kNN purity** — what fraction of a class's 10 nearest neighbours in 256-d space share the same label?
3. **Inter-class cosine similarity** of class prototype features
4. **Confusion matrix** — what are FACEOFF/POST_WHISTLE being confused with?
5. **Per-class accuracy**

### Results: Baseline `joint` model

| Class | Accuracy | kNN Purity |
|---|---|---|
| GLID_FORW (n=2356) | 83.2% | 0.771 |
| ACCEL_FORW (n=1942) | 86.3% | 0.804 |
| GLID_BACK (n=410) | 65.9% | 0.536 |
| TRANS_F2B (n=136) | 71.3% | 0.643 |
| POST_WHISTLE (n=78) | **30.8%** | **0.182** |
| FACEOFF (n=64) | **15.6%** | **0.089** |
| PRONE (n=88) | 26.1% | 0.181 |
| **Overall** | **78.25%** | |

Random baseline for kNN purity = 1/11 = 0.09.

**Key finding: FACEOFF kNN purity = 0.089 — essentially random.** FACEOFF's 10 nearest neighbours in 256-d feature space are almost never other FACEOFF samples. The GCN produces nearly identical internal representations for FACEOFF and GLID_FORW.

**Confusion matrix:** Both FACEOFF (52% → GLID_FORW) and POST_WHISTLE (53% → GLID_FORW) are being swallowed by the dominant class.

**Feature similarity:** FACEOFF and POST_WHISTLE show cosine similarity ~0.6–0.8 with GLID_FORW prototypes in 256-d space.

**t-SNE:** FACEOFF and POST_WHISTLE have no cluster of their own — scattered throughout the GLID_FORW mass.

### Conclusion: This is a feature problem, not a CPR problem

The GCN features themselves are indistinguishable for FACEOFF/POST_WHISTLE. Even if CPR were perfect, the aux branch receives features that look like GLID_FORW — filtering them with a FACEOFF-specific CPR won't produce correct aux logits. The main branch (80% of the loss) is the bottleneck.

### Class Imbalance

```
GLID_FORW   n = 2356   (43% of test set)
ACCEL_FORW  n = 1942
FACEOFF     n = 64     (37× fewer than GLID_FORW)
POST_WHISTLE n = 78
```

The model sees 37 GLID_FORW samples for every FACEOFF. Under cross-entropy loss, predicting GLID_FORW for ambiguous samples is rational — it minimises expected loss.

### Results: Weighted Sampler and Focal Loss models

Both `joint_weighted_sampler` and `joint_focal` training runs **collapsed** — predicting only MAINTAIN (class 8) for every input at every epoch examined. Overall accuracy = 2.05% = 111/5426 (exact MAINTAIN fraction).

Critically, FACEOFF kNN purity *dropped* from 0.089 (baseline) to 0.014–0.020 under rebalancing — the features got worse, not better. Naive class rebalancing destabilizes this model; the 37:1 imbalance is too extreme for weighted sampling or focal loss to handle without causing mode collapse.

---

## 9. Why FACEOFF Was Hard in the First Place

FACEOFF is not merely a hard class because of class imbalance or flat CPR — there are several structural reasons why the model as designed was unlikely to perform well on it regardless of those factors. Understanding these is important for setting realistic expectations.

### 9.1 The Skeleton Alone Is Ambiguous

A FACEOFF and a slow GLID_FORW are both a player in a relatively upright stance, weight balanced over both skates, gliding at low speed. The distinguishing features of a faceoff — the slight forward crouch, the stick held vertically with the blade on the ice — are subtle in 2D xy coordinates. The difference in joint positions between the two postures may be less than the natural variation within the GLID_FORW class due to different player body types.

The diagnostic confirmed this: FACEOFF kNN purity in 256-d feature space was 0.089 — essentially random — even though the model has 9 deep GCN layers processing the skeleton. The model genuinely cannot form separable FACEOFF features from raw joint coordinates.

### 9.2 FACEOFF Is Defined by Context, Not Pose

In hockey, what makes a moment a FACEOFF is not primarily what the player's body looks like. It is:

- **The puck has not been dropped yet** — the player is stationary and positioned at a face-off dot
- **An opponent is facing them** — the relative position/orientation between two players is the defining signal
- **The rink location** — faceoffs always happen at one of five specific locations on the ice

All three of these signals are unavailable to the model:
- The puck is not in the skeleton data
- We only have one player's skeleton (`M=1`) — the opposing player is absent
- Rink position was removed by hip normalization (by design, to aid generalization)

This means FACEOFF is being classified purely from the body posture of a single player during a brief moment of stillness — which looks nearly identical to any other moment when a player happens to be skating slowly or pausing.

### 9.3 The Class Is Transitional and Brief

FACEOFF clips capture the moment a player assumes the faceoff stance. This is brief (~1–2 seconds in real time). After temporal crop and resize to 64 frames, the distinctive portion of the clip (the final approach and stillness) may be spread thinly or randomly positioned depending on the crop. The model has no mechanism to attend to a specific moment in the clip — temporal pooling averages over everything.

POST_WHISTLE has the same property: it captures the first few seconds after a whistle, when the player transitions from active skating to passive gliding. Most of the clip still looks like gliding.

### 9.4 Class Imbalance Compounds Everything

With only 64 FACEOFF samples vs. 2356 GLID_FORW in the test set (37:1 ratio), any ambiguous sample is correctly classified by the model as GLID_FORW under cross-entropy loss. This is not a failure of the model — it is rational loss minimisation. Given that the skeleton-level features are genuinely similar, predicting GLID_FORW for uncertain inputs is the highest-expected-accuracy strategy. Attempts to correct this (weighted sampler, focal loss) caused mode collapse because the imbalance was too extreme for those methods to stabilize training.

### 9.5 CPR Cannot Compensate for Missing Context

Even if CPR[FACEOFF] were perfectly distinctive (high contrast, strong joint-pair clusters), the aux branch would still fail because it receives features from the main GCN path — features that look like GLID_FORW. Multiplying GLID_FORW-like features by a FACEOFF-specific CPR matrix produces FACEOFF-filtered GLID_FORW features. The aux logit for FACEOFF will still be low. The text prior can only amplify existing discriminative signal; it cannot create signal that is not present in the skeleton.

---

## 10. Known Implementation Issues

The following are concrete limitations in the current LAGCN implementation that could be addressed.

### 10.1 Frozen CPR Cannot Adapt to Wrong Priors

CPR is computed once and frozen (`requires_grad=False`). For classes where the text prior is accurate (GLID_FORW, ACCEL_FORW), this is fine. For classes where the text produces a flat or incorrect prior (FACEOFF, POST_WHISTLE), the frozen CPR actively adds noise to gradients throughout training. There is no mechanism to reduce its influence or correct it during training.

**Fix:** Make CPR learnable for flat-prior classes, initialized from text but allowed to drift toward whatever adjacency structure is actually discriminative. Controlled by a per-class `requires_grad` mask based on CPR discriminability score.

### 10.2 Uniform Aux Loss Weight Across Classes

The 0.2 aux weight is applied uniformly to every sample regardless of which class it belongs to. FACEOFF and POST_WHISTLE get the same aux gradient as GLID_FORW, even though their CPR provides no useful signal. This means the aux branch is adding noise to gradients for the hardest classes while providing signal for the easy ones.

**Fix:** Weight the aux loss contribution per sample by the discriminability of that sample's true class's CPR matrix. Classes with flat CPR get near-zero aux weight; classes with rich CPR get full weight.

### 10.3 Aux Branch Capacity Is Very Small

The entire CPR classification head is a single `Conv2d(256, 1, kernel_size=1)`. This applies an identical 256→1 linear projection across all 20 joint positions and all 11 class dimensions simultaneously. It cannot learn different projections for different joints or different classes. A FACEOFF-relevant joint dimension and a GLID_FORW-relevant joint dimension are compressed by the same learned weights.

**Fix:** A small per-class MLP, or separate conv weights per class, would allow the head to learn which channel dimensions are diagnostic for each class independently.

### 10.4 Joint Mean-Pooling Discards Spatial Discrimination

The last step of the aux branch is `aux_x.mean(2)` — averaging over all 20 joints to get a single logit per class. This means a model where only 2–3 joints have discriminative signal for a given class gets that signal diluted by 17–18 non-informative joints.

**Fix:** Replace the mean with a learned attention over joints — a softmax-weighted sum where the attention weights are learned per class. This lets the model learn "which joints matter for each class."

### 10.5 Hip Normalization Removes a Discriminative Feature for FACEOFF

Hip-center subtraction removes all rink position information. For most classes this is correct — the action should be rink-position-invariant. But FACEOFF always occurs at one of five specific rink locations (face-off dots). Rink position is actually a strong discriminative signal for FACEOFF that the normalization throws away. This is a fundamental trade-off: normalization helps generalization for all other classes but specifically hurts FACEOFF.

There is no easy fix without rethinking the normalization strategy per class, which is architecturally complex.

### 10.6 Single-Person Skeleton Misses Opponent Context

The dataset uses `M=1` (single person). FACEOFF is inherently a two-player event — the player is facing an opponent in a specific stance. The relative orientation between two players is a strong discriminative signal that is completely absent from the input. Even if the model were otherwise perfect, it is missing a key input modality.

### 10.7 2D vs 3D Coordinates

Hockey uses 2D keypoints (xy only). Depth information is lost. A FACEOFF crouch in 3D (slight forward lean, spine angle, stick blade angle) may look nearly identical to a slow glide in 2D projection from a typical arena camera angle.

---

## 11. BERT vs CLIP for CPR Generation

### Why BERT Is Suboptimal for CPR

The baseline hockey CPR uses `bert-base-uncased` with `pooler_output` as the sentence embedding. Several properties of BERT make it a poor fit for this task:

**1. `pooler_output` is not a semantic similarity representation.**
BERT's pooler output is the `[CLS]` token after a linear+tanh layer, designed for next-sentence prediction during pretraining. It is not optimised for measuring semantic similarity between sentences. Using it for cosine similarity comparisons between joint descriptions is technically incorrect — it can produce high similarity between unrelated sentences. Better BERT-based options (e.g. Sentence-BERT) are specifically fine-tuned for this.

**2. BERT has no grounding in physical/visual concepts.**
BERT is trained on text corpora (Wikipedia, BooksCorpus). It understands the statistical co-occurrence of words but has no representation of what a joint *looks like* or how it *moves*. "Left knee" and "right knee" will be near-synonyms in BERT's embedding space — their physical distinction (opposite sides of the body) is not captured by text co-occurrence.

**3. The template sentences are unnatural.**
`"when faceoff left knee of hockey player."` is not a natural English sentence. BERT's tokenizer and encoder are optimised for natural prose. Unusual phrasing can produce unstable or poor representations. Importantly, all 11 class templates follow the same grammatical pattern, differing only in the class name — BERT may produce similar representations for all of them if the syntactic frame dominates over the semantic content.

**4. Joint names may be out-of-distribution.**
Joint names like `"stick top"`, `"stick mid"`, `"stick tip"` are domain-specific. BERT, trained on general text, has seen minimal use of these terms in meaningful context and will produce poor embeddings for them.

### Why CLIP Is Better

CLIP (`ViT-B/32`) is trained with contrastive learning to align text representations with visual image representations. Several properties make it more suitable:

**1. CLIP embeddings are grounded in visual concepts.**
Because CLIP's text encoder was trained to match image patches, its text representations capture visual and physical properties — what something *looks like*, not just what words co-occur. "Left knee bent" and "left knee extended" will be more separated in CLIP space than BERT space.

**2. CLIP handles sport and body descriptions better.**
CLIP's training data includes image captions describing human poses, sports, and body positions. "A hockey player crouching with their stick on the ice" is the kind of phrase that appeared in CLIP training data aligned with actual images of that pose.

**3. Better sentence-level discrimination.**
CLIP's text encoder produces sentence-level representations that vary more strongly with semantic content — making it more likely that `"faceoff left elbow"` and `"gliding left elbow"` produce meaningfully different vectors.

### Available CLIP CPR Matrices

Three CLIP-based CPR matrices have been generated and stored in `graph/cls_matrix_hockey/`:

| File | Template |
|---|---|
| `clip_natural_[C]_[J].npy` | Short natural descriptions per class + joint |
| `clip_pasta_grounded_[C]_[J]-with-punctuation.npy` | PASTA body-part descriptions + joint |
| `clip_when_[C]_[J]_of_hockey_player.npy` | `"when [C] [J] of hockey player"` template with CLIP |

Corresponding training configs: `configs/hockey/joint_clip_pasta.yaml` and `configs/hockey/joint_clip_natural.yaml`. Both runs are already trained (checkpoints up to epoch 65 in `work_dir/hockey/joint_clip_natural_CUDNN/` and `work_dir/hockey/joint_clip_pasta_CUDNN/`).

The key question is whether CLIP CPR matrices show higher contrast and better inter-class discrimination for FACEOFF and POST_WHISTLE compared to BERT. This should be visualised with `visualize_cpr.py` using the CLIP topo strings before drawing conclusions from the trained models.

### Expected Difference

Even with CLIP, the core structural problems (single-person skeleton, missing rink position, genuine skeleton ambiguity) remain. However, if CLIP produces a flatter intra-class CPR and more distinct inter-class CPRs for FACEOFF, the aux branch will at least stop adding noise and may start contributing a small positive signal. The diagnostic on the CLIP-trained models (kNN purity, confusion matrix) will reveal whether the better CPR translates to better learned features.

---

## 12. Experiment Results

### 12.1 Temporal CPR vs Baseline

| Class | Baseline | Temporal CPR | Δ |
|---|---|---|---|
| GLID_FORW | 83.2% / 0.771 | 83.6% / 0.769 | ≈0 |
| ACCEL_FORW | 86.3% / 0.804 | 87.0% / 0.805 | +0.7pp |
| GLID_BACK | 65.9% / 0.536 | 64.4% / 0.523 | -1.5pp |
| POST_WHISTLE | 30.8% / 0.182 | **17.9% / 0.150** | **-12.9pp** |
| FACEOFF | 15.6% / 0.089 | 14.1% / 0.070 | -1.5pp |
| **Overall** | **78.25%** | **78.05%** | -0.2pp |

**Result: no meaningful improvement.** Temporal CPR held overall accuracy flat but hurt POST_WHISTLE badly (-12.9pp). The explanation is confirmed by the theory: temporal CPR can only amplify discriminative signal that already exists in the per-frame features. Since CPR[FACEOFF] and CPR[POST_WHISTLE] are nearly flat/uniform, applying them per-timestep vs. once on the time-average makes no practical difference — the CPR weights are still uniform so no temporal moment is emphasized over another. The POST_WHISTLE regression suggests that at certain timesteps the joint pattern briefly resembles another class's CPR, and that noise accumulates when pooled over 16 frames of CPR-filtered activations.

**Conclusion:** Temporal CPR is only useful in conjunction with either better CPR matrices (less flat for event classes) or better input features — not as a standalone change.

### 12.2 Velocity Stream vs Baseline

| Class | Baseline | Velocity stream | Δ |
|---|---|---|---|
| GLID_FORW | 83.2% / 0.771 | 83.5% / 0.729 | ≈0 |
| ACCEL_FORW | 86.3% / 0.804 | 85.3% / 0.782 | -1.0pp |
| GLID_BACK | 65.9% / 0.536 | **45.9% / 0.377** | **-20pp** |
| ACCEL_BACK | 45.0% / 0.414 | **31.5% / 0.289** | **-13.5pp** |
| TRANS_F2B | 71.3% / 0.643 | 59.6% / 0.550 | -11.7pp |
| POST_WHISTLE | 30.8% / 0.182 | 29.5% / **0.262** | +0.5pp acc / +44% purity |
| FACEOFF | 15.6% / 0.089 | **23.4% / 0.100** | **+7.8pp** |
| **Overall** | **78.25%** | **75.76%** | **-2.5pp** |

**Result: the only change that moved FACEOFF, but at a heavy cost.** FACEOFF accuracy improved +7.8pp and POST_WHISTLE kNN purity jumped from 0.182 → 0.262 (the largest purity gain of any class). The velocity signal encodes deceleration into a faceoff stance and the stillness of post-whistle gliding — both invisible in raw coordinates. However, the direction-sensitive classes collapsed badly: GLID_BACK lost 20pp, ACCEL_BACK lost 13.5pp, TRANS_F2B lost 11.7pp. These classes discriminate forward vs. backward and acceleration direction from the trajectory of joint positions over time — information that is destroyed when you replace absolute coordinates with frame-to-frame deltas.

**Conclusion:** Velocity stream is a net loss as a standalone. It is complementary to the joint stream but cannot replace it.

### 12.3 Why Joint + Motion Fusion Is the Natural Next Step

The results reveal a clear stream-class interaction:

| Stream | Good for | Poor for |
|---|---|---|
| Joint (coordinates) | GLID_BACK, ACCEL_BACK, TRANS_F2B, TRANS_B2F | FACEOFF, POST_WHISTLE |
| Motion (velocity) | FACEOFF, POST_WHISTLE | GLID_BACK, ACCEL_BACK, TRANS_F2B |

No single stream handles all classes well because different classes are defined by different types of signal:
- **Motion-defined classes** (GLID_BACK, ACCEL_BACK, TRANS): the *direction* and *rate* of motion encodes the class. Raw coordinates carry this as trajectory.
- **Event-defined classes** (FACEOFF, POST_WHISTLE): the class is defined by a specific *moment of transition* — deceleration into stillness, or drift after a whistle. Velocity directly encodes this; coordinates do not.

Running both streams through independent backbones and fusing their `(N*M, 256, 16, 20)` feature maps allows the model to use whichever stream is more informative for each class. A gated fusion mechanism learns this class-stream interaction end-to-end: for GLID_BACK samples it learns to weight the joint stream heavily; for FACEOFF samples it learns to weight the motion stream.

The CPR aux branch remains on the joint stream only — exemplar matrices encode positional joint co-activation priors derived from text descriptions of body postures, which is meaningless when applied to velocity features.

---

### 12.4 Joint + Motion Fusion Results

**Run:** `work_dir/hockey/joint_motion_fusion_CUDNN` | 65 epochs | LR schedule: 0.1 → 0.01 at epoch 35 → 0.001 at epoch 55

#### Overall

| Metric | Baseline | Fusion (best, ep57) | Fusion (final, ep65) |
|---|---|---|---|
| Top1 | 78.25% | **80.34%** (+2.09pp) | 79.34% (+1.09pp) |
| Top3 | — | 95.36% | 95.06% |
| Top5 | — | 98.62% | 98.29% |
| Mean class acc | — | **59.05%** | 56.91% |
| Train acc | — | — | 91.64% |

#### Per-class at best checkpoint (epoch 57)

| Class | Baseline | Fusion (ep57) | Δ |
|---|---|---|---|
| GLID_FORW | 83.2% | 85.61% | +2.4pp |
| ACCEL_FORW | 86.3% | 87.90% | +1.6pp |
| GLID_BACK | 65.9% | 65.37% | ≈0 |
| ACCEL_BACK | 45.0% | 43.24% | -1.8pp |
| TRANS_F2B | 71.3% | 63.24% | -8.1pp |
| TRANS_B2F | — | 42.37% | — |
| **POST_WHISTLE** | **30.8%** | **52.56%** | **+21.8pp** |
| FACEOFF | 15.6% | 21.88% | +6.3pp |
| MAINTAIN | — | 87.39% | — |
| PRONE | — | 25.00% | — |
| ON_A_KNEE | — | 75.00% | — |

#### Training dynamics

The run has three distinct phases driven by LR decays:

| Phase | Epochs | LR | Test Top1 | Train acc |
|---|---|---|---|---|
| Initial | 1–35 | 0.1 | 70–77% | 73–77% |
| Post-decay 1 | 36–55 | 0.01 | 78–80% | 80–85% |
| Post-decay 2 | 56–65 | 0.001 | 79–80% | 88–92% |

Both LR decays produced an immediate jump in test accuracy. The model peaked at epoch 57 (80.34%) then began overfitting: train acc kept climbing to 91.84% while test Top1 drifted back to 79.34% by epoch 65. The train–test gap at the final epoch is ~12pp. An early stop at epoch 57 would have been optimal.

#### Interpretation

The fusion hypothesis is confirmed. **POST_WHISTLE gained +21.8pp** — the largest single-class improvement of any experiment. The motion stream correctly identified the deceleration-into-stillness signature that is invisible in raw coordinates. FACEOFF also improved +6.3pp for the same reason.

Direction-sensitive classes are a mixed picture. GLID_BACK is essentially recovered (≈0 vs. the -20pp collapse of the standalone velocity stream), confirming the gate learned to suppress the motion stream for this class. However, TRANS_F2B is still -8.1pp below baseline and ACCEL_BACK is slightly below — suggesting the gate is not fully suppressing the motion stream for all direction-sensitive classes, or the shared training signal is slightly shifting the joint backbone away from the direction-encoding representation.

**FACEOFF and PRONE remain structural limitations.** FACEOFF at 21.88% is better than baseline (15.6%) but still heavily confused with GLID_FORW in the main branch. Both classes suffer from the skeleton ambiguity identified in the CPR analysis: no additional input stream resolves the core issue that the body posture is indistinguishable from gliding at the GCN feature level.

---

### 12.5 Data Split Fix (combined_annotations_v2)

Before continuing with new runs, the test split was found to be incorrect. The original `test.pkl` (Jan 7) had only 64 FACEOFF test samples; the correct v2 split has 111. The v2 split files were generated from `combined_annotations_v2.pkl` and stored locally:

All configs updated to point to `data/annotations/`. All results from section 12.6 onward use the corrected split.

---

### 12.6 Fusion + T=30 Window (correct split, no accel)

**Run:** `work_dir/hockey/joint_motion_fusion_w30_CUDNN` | 65 epochs | correct v2 split | `window_size=30`, `p_interval=[0.75,1]`, `motion_in_channels=2`

Key changes from section 12.4: window reduced from 64→30 (matching dataset max), crop lower bound raised 0.5→0.75 to avoid cropping out transition moments.

| Metric | Value |
|---|---|
| **Best Top1 (ep37)** | **80.92%** |
| Mean class acc (ep37) | 55.37% |
| Final epoch (65) Top1 | 79.31% |
| Train acc (final) | 96.53% |

Per-class at best epoch (37):

| Class | Acc | n |
|---|---|---|
| GLID_FORW | 87.1% | 2356 |
| ACCEL_FORW | 87.8% | 1942 |
| GLID_BACK | 65.9% | 410 |
| ACCEL_BACK | 39.6% | 111 |
| TRANS_F2B | 66.9% | 136 |
| TRANS_B2F | 37.3% | 118 |
| POST_WHISTLE | 48.7% | 78 |
| **FACEOFF** | **71.2%** | 111 |
| MAINTAIN | 28.4% | 88 |
| PRONE | 58.3% | 12 |
| ON_A_KNEE | 0.0% | 21 |

FACEOFF jumping to 71.2% (from 21.9% on old split) is the direct effect of the corrected training data — the new train split has 702 FACEOFF samples vs far fewer in the old split. Best epoch is 37 (immediately after first LR decay at step 35). The model peaks early and overfits heavily in phase 3 (train 96.5% vs test 79.3% at epoch 65, a 17pp gap).

---

### 12.7 Fusion + T=30 + Acceleration Channel

**Run:** `work_dir/hockey/joint_motion_fusion_w30_accel_CUDNN` | 65 epochs | correct v2 split | `window_size=30`, `p_interval=[0.75,1]`, `motion_in_channels=4`

New addition: acceleration stream (second-order delta) concatenated as channels 3–4 of the motion stream. The motion tower now receives `(N, 4, T, V, M)` — channels 1–2 are frame-to-frame velocity (Δxy), channels 3–4 are the delta of velocity (Δ²xy). Only `motion_in_channels` in `FusionModel` and the feeder's `dual_stream` block changed; architecture is otherwise identical.

#### Overall

| Metric | w30 no-accel (ep37) | **w30 + accel (ep39)** | Δ |
|---|---|---|---|
| **Best Top1** | 80.92% | **81.37%** | +0.45pp |
| Mean class acc | 55.37% | 58.36% | +2.99pp |
| Final epoch Top1 | 79.31% | 79.42% | ≈0 |
| Train acc (final) | 96.53% | 95.94% | ≈0 |

#### Per-class at best epoch (39)

| Class | w30 no-accel | w30 + accel | Δ | n |
|---|---|---|---|---|
| GLID_FORW | 87.1% | 86.3% | -0.8pp | 2356 |
| ACCEL_FORW | 87.8% | 88.2% | +0.4pp | 1942 |
| GLID_BACK | 65.9% | 64.9% | -1.0pp | 410 |
| **ACCEL_BACK** | **39.6%** | **58.6%** | **+19.0pp** | 111 |
| **TRANS_F2B** | **66.9%** | **72.1%** | **+5.2pp** | 136 |
| TRANS_B2F | 37.3% | 44.9% | +7.6pp | 118 |
| POST_WHISTLE | 48.7% | 34.6% | -14.1pp | 78 |
| **FACEOFF** | **71.2%** | **82.9%** | **+11.7pp** | 111 |
| MAINTAIN | 28.4% | 25.0% | -3.4pp | 88 |
| PRONE | 58.3% | 75.0% | +16.7pp | 12 |
| ON_A_KNEE | 0.0% | 9.5% | +9.5pp | 21 |

#### Interpretation

**ACCEL_BACK +19pp** is the clearest validation of the acceleration channel. ACCEL_BACK requires distinguishing active push-off (increasing velocity) from steady backward glide — the second-order delta directly encodes this as a large positive Δ²xy during push-off and near-zero during glide. ACCEL_FORW, TRANS_F2B, and TRANS_B2F also improved, consistent with the acceleration signal being discriminative wherever a velocity transition occurs.

**FACEOFF +11.7pp (82.9%)** — the deceleration-into-stance signature is strongly captured. With 702 training samples and the accel signal, the model reliably identifies the approach phase.

**POST_WHISTLE -14.1pp is the key regression.** 44 of 78 POST_WHISTLE samples are predicted as GLID_FORW. Post-whistle gliding has near-zero second derivative (passive slowdown), which overlaps with near-zero acceleration of steady gliding. The channel that helps ACCEL_BACK actively hurts POST_WHISTLE — both have a specific acceleration signature, but POST_WHISTLE's is too similar to gliding at the Δ² level.

**MAINTAIN (25%) and ON_A_KNEE (9.5%)** remain limited by data starvation: 442 and 39 training samples respectively. No stream engineering fixes this.

**TRANS_B2F (44.9%) is structurally harder than TRANS_F2B (72.1%)** — the B2F transition exits into forward acceleration, making the clip's final frames indistinguishable from ACCEL_FORW. 40 of 118 TRANS_B2F samples are predicted as ACCEL_FORW. F2B exits into backward skating, which is less common and more distinctive.

**Precision vs recall asymmetry on FACEOFF:** recall=82.9% but precision=49.5% (92/186 correct predictions). The model triggers on low-crouch postures from ACCEL_FORW and MAINTAIN as false FACEOFF detections.

#### Remaining improvement opportunities

| Class | Issue | Direction |
|---|---|---|
| POST_WHISTLE (34.6%) | Accel channel creates false GLID_FORW overlap | Speed magnitude feature; per-class gate weighting |
| TRANS_B2F (44.9%) | Exit phase identical to ACCEL_FORW | Longer clips or better annotation boundaries |
| MAINTAIN (25.0%) | Overlaps GLID_FORW and FACEOFF | More training data |
| GLID_BACK (64.9%) | Forward/backward ambiguity at T=30 | Longer temporal window |
| FACEOFF precision (49.5%) | False triggers from other low-crouch postures | Hard negative mining |
| ON_A_KNEE (9.5%) | 39 training samples | More data only |

---

### 12.8 Imbalance Strategies + Sequence Modelling Postprocessing

Three new experiments built on top of the 12.7 baseline (`w30 + accel`, ep39, Top1=81.37%, MCA=58.36%).

---

#### 12.8.1 Balanced Batch Sampler

**Run:** `work_dir/hockey/joint_motion_fusion_w30_accel_balanced_CUDNN` | same hyperparams as 12.7 | `batch_size=55` (5 samples × 11 classes per batch)

Each training batch is forced to contain exactly 5 samples from every class via `BalancedBatchSampler` (`feeders/balanced_sampler.py`). Rare classes (PRONE, ON_A_KNEE, ACCEL_BACK) that would appear ~0–1 times per standard batch now appear exactly 5 times. The sampler reshuffles within each class when exhausted and stops after approximately one epoch of data.

Config: `configs/hockey/joint_motion_fusion_w30_accel_balanced.yaml` — `use_balanced_sampler: true`, `batch_size: 55`.

**Best epoch: 59** | Top1=78.64% | MCA=53.31%

#### Per-class at best epoch (59) vs 12.7 baseline

| Class | Baseline (ep39) | Balanced sampler (ep59) | Δ | n |
|---|---|---|---|---|
| GLID_FORW | 86.3% | 83.8% | -2.5pp | 2356 |
| ACCEL_FORW | 88.2% | 87.6% | -0.6pp | 1942 |
| GLID_BACK | 64.9% | 61.5% | -3.4pp | 410 |
| ACCEL_BACK | 58.6% | 38.7% | -19.9pp | 111 |
| TRANS_FORW_TO_BACK | 72.1% | 61.8% | -10.3pp | 136 |
| TRANS_BACK_TO_FORW | 44.9% | 28.8% | -16.1pp | 118 |
| POST_WHISTLE_GLIDING | 34.6% | 26.9% | -7.7pp | 78 |
| FACEOFF_BODY_POSITION | 82.9% | 79.3% | -3.6pp | 111 |
| MAINTAIN_POSITION | 25.0% | 23.9% | -1.1pp | 88 |
| PRONE | 75.0% | 75.0% | 0.0pp | 12 |
| ON_A_KNEE | 9.5% | 19.1% | +9.6pp | 21 |

**Overall:**

| Metric | Baseline (ep39) | Balanced sampler (ep59) | Δ |
|---|---|---|---|
| Top-1 | 81.37% | 78.64% | -2.73pp |
| Mean class acc | 58.36% | 53.31% | -5.05pp |

**Interpretation:** The balanced batch sampler underperformed both the baseline and the weighted loss run. Forcing exactly 5 samples per class per batch severely over-represents the rarest classes (PRONE: 12 train samples, ON_A_KNEE: 39) relative to the dominant ones, creating gradient instability — the model sees PRONE as frequently as GLID_FORW despite a ~180× difference in actual class frequency. ON_A_KNEE improves (+9.6pp) as expected, but nearly every other class regresses, including ACCEL_BACK (-19.9pp) and both transition classes. The weighted loss approach (12.8.2) is a strictly better strategy for this dataset: it re-weights the loss signal without distorting the data distribution the model learns from.

---

#### 12.8.2 Weighted Cross-Entropy Loss

**Run:** `work_dir/hockey/joint_motion_fusion_w30_accel_weighted_cls_CUDNN` | same hyperparams as 12.7 | `batch_size=32`, standard shuffle

Per-class weights applied to the main `CrossEntropyLoss` (aux branch remains unweighted — CPR is a structural prior). Weights are inverse-frequency derived:

| Class | Weight |
|---|---|
| GLID_FORW | 0.2162 |
| ACCEL_FORW | 0.2162 |
| GLID_BACK | 0.2480 |
| ACCEL_BACK | 0.5217 |
| TRANS_FORW_TO_BACK | 0.4594 |
| TRANS_BACK_TO_FORW | 0.4763 |
| POST_WHISTLE_GLIDING | 0.5372 |
| FACEOFF_BODY_POSITION | 0.3699 |
| MAINTAIN_POSITION | 0.4915 |
| PRONE | 2.9596 |
| ON_A_KNEE | 3.6038 |

Config: `configs/hockey/joint_motion_fusion_w30_accel_weighted_cls.yaml` — `class_weights: [...]`.

**Best epoch: 38** | Top1=79.40% | MCA=63.32%

#### Per-class at best epoch (38) vs 12.7 baseline

| Class | Baseline (ep39) | Weighted cls (ep38) | Δ | n |
|---|---|---|---|---|
| GLID_FORW | 86.3% | 83.2% | -3.1pp | 2356 |
| ACCEL_FORW | 88.2% | 85.2% | -3.0pp | 1942 |
| GLID_BACK | 64.9% | 61.7% | -3.2pp | 410 |
| ACCEL_BACK | 58.6% | 56.8% | -1.8pp | 111 |
| TRANS_FORW_TO_BACK | 72.1% | 73.5% | +1.4pp | 136 |
| TRANS_BACK_TO_FORW | 44.9% | 50.9% | **+6.0pp** | 118 |
| POST_WHISTLE_GLIDING | 34.6% | 55.1% | **+20.5pp** | 78 |
| FACEOFF_BODY_POSITION | 82.9% | 77.5% | -5.4pp | 111 |
| MAINTAIN_POSITION | 25.0% | 42.1% | **+17.1pp** | 88 |
| PRONE | 75.0% | 58.3% | -16.7pp | 12 |
| ON_A_KNEE | 9.5% | 52.4% | **+42.9pp** | 21 |

**Overall:**

| Metric | Baseline (ep39) | Weighted cls (ep38) | Δ |
|---|---|---|---|
| Top-1 | 81.37% | 79.40% | -1.97pp |
| **Mean class acc** | 58.36% | **63.32%** | **+4.96pp** |

**Interpretation:** The class weights trade overall Top-1 accuracy for significantly better coverage of minority and difficult classes. ON_A_KNEE jumps from 9.5% → 52.4% (+42.9pp), MAINTAIN from 25% → 42.1% (+17.1pp), POST_WHISTLE from 34.6% → 55.1% (+20.5pp). The cost is paid by the dominant classes (GLID_FORW, ACCEL_FORW, FACEOFF). PRONE regresses (-16.7pp) likely because its 12 test samples are highly sensitive to small shifts and PRONE has similar posture to ON_A_KNEE. This is the best MCA result to date.

---

#### 12.8.3 Viterbi Sequence Modelling (Postprocessing)

Postprocessing script: `sequence_modelling.py`. Uses a pre-computed transition probability matrix (`transition_probs.csv`, estimated from train sequences in `sequence_annotations.pkl`) to run Viterbi decoding across each tracklet's ordered action sequence. `TRANSITION_WEIGHT=0.1` — 10% transition contribution, 90% emission (model scores).

The transition matrix enforces domain constraints (e.g. GLID_FORW → GLID_BACK is disallowed; direction must pass through a transition class) and applies smoothed empirical probabilities for allowed transitions.

Pipeline to reproduce:
```bash
# 1. Remap score pkl keys from feeder-style (test_0, test_1...) to frame_dir keys
python remap_score_keys.py \
  work_dir/hockey/<run>/epoch1_test_score.pkl \
  data/annotations/test.pkl \
  work_dir/hockey/<run>/epoch<N>_test_score_remapped.pkl

# 2. Run Viterbi adjustment
python sequence_modelling.py \
  --results work_dir/hockey/<run>/epoch<N>_test_score_remapped.pkl \
  --transition-csv transition_probs.csv

# 3. Evaluate
python eval_sequence_adjustment.py  # edit paths inside the script
```

**Results on baseline w30+accel (ep39):**

| Metric | Original | Viterbi (w=0.1) | Δ |
|---|---|---|---|
| Top-1 | 81.37% | 82.41% | **+1.04pp** |
| Mean class acc | 58.36% | 57.34% | -1.01pp |

| Class | Original | Viterbi | Δ |
|---|---|---|---|
| GLID_FORW | 86.3% | 89.1% | +2.8pp |
| ACCEL_FORW | 88.2% | 88.7% | +0.5pp |
| GLID_BACK | 64.9% | 63.4% | -1.5pp |
| ACCEL_BACK | 58.6% | 58.6% | 0.0pp |
| TRANS_FORW_TO_BACK | 72.1% | 66.2% | -5.9pp |
| TRANS_BACK_TO_FORW | 44.9% | 37.3% | -7.6pp |
| POST_WHISTLE_GLIDING | 34.6% | 41.0% | +6.4pp |
| FACEOFF_BODY_POSITION | 82.9% | 82.9% | 0.0pp |
| MAINTAIN_POSITION | 25.0% | 23.9% | -1.1pp |
| PRONE | 75.0% | 75.0% | 0.0pp |
| ON_A_KNEE | 9.5% | 4.8% | -4.8pp |

**Results on weighted cls (ep38):**

| Metric | Original | Viterbi (w=0.1) | Δ |
|---|---|---|---|
| Top-1 | 79.40% | 81.01% | **+1.62pp** |
| Mean class acc | 63.32% | 62.28% | -1.04pp |

| Class | Original | Viterbi | Δ |
|---|---|---|---|
| GLID_FORW | 83.2% | 86.6% | +3.4pp |
| ACCEL_FORW | 85.2% | 86.7% | +1.5pp |
| GLID_BACK | 61.7% | 60.7% | -1.0pp |
| ACCEL_BACK | 56.8% | 59.5% | +2.7pp |
| TRANS_FORW_TO_BACK | 73.5% | 64.7% | -8.8pp |
| TRANS_BACK_TO_FORW | 50.9% | 41.5% | -9.3pp |
| POST_WHISTLE_GLIDING | 55.1% | 51.3% | -3.9pp |
| FACEOFF_BODY_POSITION | 77.5% | 80.2% | +2.7pp |
| MAINTAIN_POSITION | 42.1% | 43.2% | +1.1pp |
| PRONE | 58.3% | 58.3% | 0.0pp |
| ON_A_KNEE | 52.4% | 52.4% | 0.0pp |

**Interpretation:** Viterbi consistently improves Top-1 (+1.0–1.6pp) by smoothing dominant-class predictions across sequences. However, it hurts mean class accuracy because the transition matrix is estimated from training data and biases against rare transitions — TRANS_BACK_TO_FORW loses -7.6 to -9.3pp across both models. The transition weight of 0.1 is already conservative; lower values or a no-constraints matrix may help retain minority-class gains.

---

## 13. Joint + Motion Fusion Implementation

### Architecture

```
Joint input  (N, 2, 30, 20, 1) ──→ joint_l1...joint_l9 ──→ (N*M, 256, 8, 20) ──┐
                                                                                   ├─→ StreamFusion(gate) ─→ (N*M, 256, 8, 20) ─→ FC ─→ main logits
Motion input (N, 4, 30, 20, 1) ──→ motion_l1...motion_l9 ─→ (N*M, 256, 8, 20) ──┘
                    ↑
           joint features also feed CPR aux branch ─→ aux logits (N, 11)
```

Note: `window_size` was originally 64 (T'=16 after two stride-2 layers) and was changed to 30 (T'=8), matching the dataset maximum frame count and eliminating redundant bilinear upsampling. The motion stream was originally 2 channels (velocity Δxy only) and was extended to 4 channels (velocity Δxy + acceleration Δ²xy); see section 12.7.

**StreamFusion (gate mode):**
```
gate = sigmoid(BN(ReLU(Conv2d(512→256, 1×1))) applied to cat([joint, motion]))
output = gate × joint_features + (1 − gate) × motion_features
```
The gate learns a per-channel blend ratio. For GLID_BACK, the gate saturates toward the joint stream; for FACEOFF, toward the motion stream. This is learned automatically from the classification loss — no explicit supervision on stream selection is needed.

### Files

| File | Change |
|---|---|
| `model/lagcn_fusion.py` | New — `StreamFusion` module + `FusionModel` class; `motion_in_channels` param added to support different channel counts per tower |
| `feeders/feeder_hockey.py` | Added `dual_stream=False` flag; returns `(joint, motion_stream)` tuple where `motion_stream` is `(4, T, V, M)` — velocity + acceleration |
| `main.py` | `self.model(*data)` dispatch when data is a tuple |
| `configs/hockey/joint_motion_fusion_w30_accel.yaml` | Active config — `window_size: 30`, `p_interval: [0.75,1]`, `dual_stream: True`, `motion_in_channels: 4`, correct v2 data paths |

`model/lagcn.py` and all existing single-stream configs are untouched.

### Dual Stream Feeder

When `dual_stream=True`, the feeder applies normalization and temporal crop once, then derives both the velocity and acceleration streams internally:

```python
data_motion = data_numpy.copy()
data_motion[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]  # velocity (Δxy)
data_motion[:, -1] = 0

data_accel = np.zeros_like(data_motion)
data_accel[:, 1:-1] = data_motion[:, 2:] - data_motion[:, 1:-1]  # acceleration (Δ²xy)

motion_stream = np.concatenate([data_motion, data_accel], axis=0)  # (4, T, V, M)
return (data_numpy, motion_stream), label, index
```

Both streams see identical crop and normalization. Velocity is derived from already-processed joint positions; acceleration is the delta of velocity. Boundary frames (first and last) are zeroed for the acceleration channel. The resulting motion stream is `(4, T, V, M)` — channels 1–2 are Δxy, channels 3–4 are Δ²xy.

### Training

Run:
```bash
python main.py --config configs/hockey/joint_motion_fusion.yaml
```

To warm-start from pretrained single-stream checkpoints (recommended):
```python
model.load_pretrained_tower('work_dir/hockey/joint_CUDNN/runs-65-20800.pt', 'joint_tower')
model.load_pretrained_tower('work_dir/hockey/joint_motion_CUDNN/runs-65-20800.pt', 'motion_tower')
```
The `StreamFusion` gate is the only component that trains from scratch. The backbone towers fine-tune from their respective pretrained weights.

### Running Experiments

```bash
# Velocity stream (joint motion)
python main.py --config configs/hockey/joint_motion.yaml

# Temporal CPR
python main.py --config configs/hockey/joint_temporal_cpr.yaml

# Joint + motion fusion with T=30 and acceleration channel (current best)
python main.py --config configs/hockey/joint_motion_fusion_w30_accel.yaml

# Diagnostic on any trained model (supports dual-stream checkpoints)
python diagnose_features.py \
    --config configs/hockey/<config>.yaml \
    --checkpoint work_dir/hockey/<run>/runs-<epoch>-<step>.pt \
    --out viz_diagnostic_<name>
```

---

## Appendix: Key Files

| File | Purpose |
|---|---|
| `model/lagcn.py` | Full model: 9 TCN-GCN layers, main branch, aux branch with CPR |
| `graph/cls_examplar.py` | Loads pre-computed CPR matrices from `.npy` files |
| `graph/cls_matrix_hockey/` | Pre-computed CPR matrices for all hockey templates |
| `graph_gen/generate_hockey_exemplars.py` | BERT-based CPR generation for hockey |
| `graph_gen/generate_hockey_exemplars_clip.py` | CLIP-based CPR generation for hockey |
| `feeders/feeder_hockey.py` | Data loading, normalization, temporal crop/resize |
| `configs/hockey/joint.yaml` | Baseline joint config |
| `configs/hockey/joint_motion.yaml` | Velocity stream config |
| `configs/hockey/joint_temporal_cpr.yaml` | Temporal CPR config |
| `configs/hockey/joint_clip_pasta.yaml` | CLIP PASTA-grounded CPR config |
| `configs/hockey/joint_clip_natural.yaml` | CLIP natural description CPR config |
| `configs/hockey/joint_motion_fusion_w30_accel.yaml` | Active config — T=30, velocity+accel motion stream, v2 data split |
| `configs/hockey/joint_motion_fusion.yaml` | Earlier fusion config (T=30, velocity only, kept for reference) |
| `model/lagcn_fusion.py` | Two-tower fusion model with gated stream fusion; `motion_in_channels` controls motion tower input depth |
| `diagnose_features.py` | Feature-space diagnostic: t-SNE, kNN purity, confusion matrix |
| `visualize_cpr.py` | CPR visualization: heatmaps, PCA, inter-class similarity |
| `LAGCN_CPR_NOTES.md` | This document |
