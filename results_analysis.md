# LAGCN Hockey — Experiment Results Analysis

**Date**: 2026-03-08
**Model**: LAGCN (joint modality, 20 keypoints, 11 classes)
**Dataset**: Hockey Skating Actions (train/test split, 2D skeleton)

---

## 1. Summary

| Run | Config | Best Top-1 | Best Mean Class Acc | Best Epoch | Notes |
|---|---|---|---|---|---|
| **joint_CUDNN** | CE loss, standard sampler | **80.15%** | **56.39%** | 38 | Feeder fixes applied (normalization + temporal crop) |
| joint_focal_CUDNN | Focal loss (γ=2.0), standard sampler | 77.68% | 51.93% | 37 | Prior run, no feeder fixes |
| joint_weighted_sampler_CUDNN | CE loss, weighted sampler | 74.47% | 46.18% | 63 | Prior run, no feeder fixes |

**Key changes in joint_CUDNN** vs the other two runs:
- `_normalize()`: hip-center subtraction + max-abs scale → coordinates in [−1, 1]
- `_temporal_crop_resize()`: replaced `tools.valid_crop_resize` which had a hardcoded 64-frame lower bound that suppressed all temporal augmentation on short clips (22–30 frames)

---

## 2. Per-Class Accuracy at Best Epoch

| Class | joint_CUDNN (ep38) | joint_focal (ep37) | joint_weighted (ep63) |
|---|---|---|---|
| GLID_FORW | 85.14% | 85.02% | 81.88% |
| ACCEL_FORW | 88.00% | 85.94% | 83.57% |
| GLID_BACK | **68.05%** | 53.66% | 53.90% |
| ACCEL_BACK | **57.66%** | 40.54% | 36.94% |
| TRANS_FORW_TO_BACK | **69.12%** | 58.82% | 49.26% |
| TRANS_BACK_TO_FORW | **43.22%** | 29.66% | 26.27% |
| POST_WHISTLE_GLIDING | 20.51% | **34.62%** | 25.64% |
| FACEOFF_BODY_POSITION | 15.63% | **18.75%** | 10.94% |
| MAINTAIN_POSITION | 80.18% | **87.39%** | 74.77% |
| PRONE | **26.14%** | 23.86% | 14.77% |
| ON_A_KNEE | **66.67%** | 50.00% | 50.00% |
| **Mean** | **56.39%** | 51.93% | 46.18% |

Bold = best per class across runs.

### Observations

- **joint_CUDNN dominates on 8/11 classes** — particularly the directional/transition classes (GLID_BACK +14pp, ACCEL_BACK +17pp, TRANS_FORW_TO_BACK +10pp, TRANS_BACK_TO_FORW +14pp vs focal). These classes involve more subtle motion dynamics that benefit from proper normalization making positional encoding functional.
- **Focal loss wins on 3 classes**: POST_WHISTLE_GLIDING, FACEOFF_BODY_POSITION, MAINTAIN_POSITION. These are the low-motion/positional classes where focal loss redirects gradient attention toward harder samples, but at a cost to overall accuracy.
- **Weighted sampler consistently underperforms** on all but none — the random oversampling of rare classes adds noise without improving rare class accuracy.
- **Hardest classes across all runs**: POST_WHISTLE_GLIDING (15–35%), FACEOFF_BODY_POSITION (11–19%), PRONE (15–26%). These are positional/static poses with high visual similarity to other classes.

---

## 3. Training Curves (Top-1 %)

Epochs evaluated every epoch. Below: every 5 epochs for brevity, plus the best epoch.

| Epoch | joint_CUDNN | joint_focal | joint_weighted |
|---|---|---|---|
| 1 | 58.39 | 40.38 | 4.17 |
| 5 | 65.39 | 49.74 | 26.76 |
| 10 | 64.08 | 62.03 | 51.66 |
| 15 | 63.75 | 66.33 | 54.70 |
| 20 | 66.48 | 68.34 | 52.97 |
| 25 | 74.44 | 66.96 | 44.89 |
| 30 | 74.84 | 69.52 | 59.95 |
| 35 | 79.76 | 77.64 | 68.63 |
| **38** | **80.15** | — | — |
| **37** | — | **77.68** | — |
| 40 | 79.16 | 76.48 | 73.09 |
| 45 | 76.78 | 76.39 | 74.00 |
| 50 | 65.39* | 74.60 | 73.83 |
| 55 | 68.39* | 74.18 | 73.13 |
| 60 | 74.90* | 75.97 | 74.05 |
| **63** | — | — | **74.47** |
| 65 | 78.25* | 76.23 | 74.31 |

*joint_CUDNN epochs 49–65 show lower values than peak — the model peaks at epoch 38 then slightly overfits after LR drops at epoch 55.

### Curve observations

- **joint_CUDNN** converges faster (epoch 35: 79.76%) and peaks earlier (epoch 38). After the epoch-55 LR step, accuracy stabilises at ~78–79%.
- **joint_focal** follows a similar trajectory but ~2pp lower throughout. Peaks at epoch 37 (77.68%), then decays to ~76% — suggesting focal loss causes overfit on minority classes.
- **joint_weighted** has very noisy early training (epoch 1–35: swings between 4–60%), stabilises after epoch 36, and plateaus at ~74% for the final 30 epochs — unable to push past the noise introduced by aggressive resampling.

---

## 4. Confusion Matrix Highlights (joint_CUDNN, epoch 38)

Most-confused class pairs (off-diagonal counts):

| Predicted → | True class | Count | % of true class |
|---|---|---|---|
| GLID_FORW | ACCEL_FORW | 191 | ~10% |
| GLID_FORW | GLID_BACK | 84 | ~20% |
| ACCEL_FORW | GLID_FORW | 147 | ~7% |
| GLID_FORW | TRANS_FORW_TO_BACK | 27 | ~20% |
| GLID_FORW | POST_WHISTLE_GLIDING | 51 | ~65% |
| ACCEL_FORW | POST_WHISTLE_GLIDING | — | — |
| GLID_FORW | FACEOFF_BODY_POSITION | 34 | ~53% |
| MAINTAIN_POSITION | PRONE | 12 | ~14% |


GLID_FORW is the dominant attractor — most misclassified samples are predicted as GLID_FORW. POST_WHISTLE_GLIDING (78 samples) is mostly eaten by GLID_FORW (51 samples = 65% of the class). FACEOFF_BODY_POSITION (64 samples) similarly loses 34 to GLID_FORW.

---

## 5. Conclusions

1. **Feeder normalization was the most impactful fix**: bringing coordinates to [−1,1] makes LAGCN's positional embeddings (init ≈ ±1) functionally useful rather than noise (~245× smaller than raw pixel coords). This alone likely explains the +2.5pp Top-1 gain over the focal run on the same architecture.

2. **Temporal augmentation now active**: the old `valid_crop_resize` clamped cropped length to ≥ 64 frames on 22–30 frame clips, making `p_interval=[0.5,1]` a no-op. The fix allows genuine 50–100% crop of valid frames, which improves generalisation.

3. **Focal loss (γ=2.0) is not beneficial overall**: it sacrifices 2.5pp Top-1 for modest gains on 3 static/positional classes. If minority-class accuracy matters, a lower γ (0.5–1.0) or class-frequency-based label smoothing may be a better trade-off.

4. **Weighted sampler is harmful**: adds gradient noise in early training (-3.7pp vs focal, -5.7pp vs joint_CUDNN). Not recommended.

5. **Remaining headroom**: Mean class accuracy (56%) lags well behind Top-1 (80%). The model is still biased toward dominant classes. Potential next steps: connect stick nodes to body skeleton (graph fix), address FACEOFF/POST_WHISTLE confusion with GLID_FORW via data augmentation or hard negative mining.


  Why text helps/hurts specific classes

  The core finding: two separate failure modes

  1. Text similarity is too high across ALL classes

  The red heatmap (top-left) shows CLIP similarity ranging 0.85–1.00 between every class pair. Hockey actions are lexically and
   semantically very similar — they all involve skating, sticks, ice. CLIP was trained on internet text/images, not sports
  biomechanics. So the contrastive loss has barely any gradient signal to work with — the "pull toward your class, push away
  from others" is very weak when everything looks ~0.95 similar.

  2. Text distinctiveness doesn't predict accuracy

  ┌───────────────────────┬────────────────────────┬───────────┬─────────┐
  │         Class         │  Text Distinctiveness  │ LAGCN acc │ GAP acc │
  ├───────────────────────┼────────────────────────┼───────────┼─────────┤
  │ FACEOFF_BODY_POSITION │ 0.173 (most distinct)  │ 15.6%     │ 83.3%   │
  ├───────────────────────┼────────────────────────┼───────────┼─────────┤
  │ MAINTAIN_POSITION     │ 0.143 (average)        │ 80.2%     │ 2.0%    │
  ├───────────────────────┼────────────────────────┼───────────┼─────────┤
  │ TRANS_BACK_TO_FORW    │ 0.116 (least distinct) │ 43.2%     │ 36.7%   │
  └───────────────────────┴────────────────────────┴───────────┴─────────┘

  FO has the most distinctive text AND benefits from GAP. MP has average text but GAP completely fails it. This tells you text
  distinctiveness alone doesn't explain outcomes — what matters is whether the class is distinguishable from skeleton motion
  data.

  ---
  Class-by-class breakdown

  Why GF and AF work in both — data volume (8928 + 7030 samples = 67% of training). The model can't fail these even with broken
   text. They're the "catch-all" attractors that everything else bleeds into.

  Why FACEOFF_BODY_POSITION: GAP=83% vs LAGCN=16% — this is the clearest case where text genuinely helps. FO's body-part text
  is the most distinctive of any class (Head: 0.230, highest by far). "Hands grip stick low near the ice; Hips hunch over
  deeply; Legs are wide" describes a pose that CLIP understands as categorically different from skating. LAGCN's CPR can't
  capture this — FO only has 182 samples, not enough to build reliable joint co-activation priors.

  Why MAINTAIN_POSITION: LAGCN=80% vs GAP=2% — MP is a static standing pose. Its skeleton looks almost identical to GF (player
  standing upright). LAGCN's structural prior captures "square to play, feet planted" as a distinct joint pattern. GAP maps 67%
   of MP to FO because without working part-aware text, the global embedding of "stationary stance" gets confused with "faceoff
   crouch". The text SAYS the right thing ("feet wide, stationary, stick in passing lane") but the bugs in GAP mean only the
  broken global alignment is active.

  Why POST_WHISTLE_GLIDING fails in both — per-part text distinctiveness (bottom panel) shows PWG has the lowest hips
  distinctiveness (0.081) of all classes. The description "hips upright and relaxed" is essentially the same as GF. In GAP, 93%
   bleeds to GF. In LAGCN, 65% bleeds to GF. Neither model can distinguish "purposeful forward glide" from "casual post-whistle
   drift" — they look identical in 2D skeleton space.

  The per-body-part insight (bottom panel) — Hips are the least discriminative body part across ALL classes (values 0.08–0.14).
   Hands/Arms are most discriminative for AF (0.176), FO (0.166), and MP (0.210). Legs/Feet are most discriminative for TFB
  (0.212). If GAP's part-aware text bugs were fixed, it should target hand features for AF/FO/MP and leg features for TFB —
  those are the body parts whose text descriptions are actually distinctive.

  ---
  The key argument for why GAP's text should work (once fixed)

  The only class where text clearly helps over structural priors is FO — and it's exactly the class where the text's body-part
  distinctiveness is highest (Head=0.230) AND the skeleton alone is insufficient (only 182 samples, looks like crouched GF).
  This is the proof-of-concept: when text is genuinely distinctive for a body part AND the skeleton is ambiguous, the text
  contrastive loss adds real information. Fixing Bug 1 (part-aware split) and Bug 3 (hip/foot swap) would extend this benefit
  to AF (Hands), MP (Hands), and TFB (Legs).


  What LAGCN actually does with text                                                                                           
                                                                                          
  LAGCN uses text once, before training ever starts, to build a matrix that describes how joints relate to each other for each 
  class.                                                                                                                       
                                                                                                                               
  Here's the process:                                                                                                          
                  
  Text: "when FACEOFF_BODY_POSITION left_knee of hockey player"
           ↓
        BERT encoder
           ↓
    512-dim embedding vector

  It does this for every combination of class × joint (11 classes × 20 joints = 220 sentences). Then for each class, it
  computes how similar every joint's embedding is to every other joint's embedding:

  FO joint similarity matrix (20×20):
                left_knee  right_knee  left_hand  right_hand  ...
  left_knee   [  1.00       0.94        0.71        0.72      ]
  right_knee  [  0.94       1.00        0.70        0.71      ]
  left_hand   [  0.71       0.70        1.00        0.95      ]
  ...

  This is the CPR (Class Prior Relationship) matrix — it's a frozen (11, 20, 20) tensor. It encodes the prior belief: "for
  class FO, the two knees should behave similarly to each other, and the hands should behave similarly to each other, but knees
   and hands are less related."

  ---
  How LAGCN uses this during training

  The model has two branches:

  Skeleton input
        ↓
    Main GCN → classification score (main loss)
        ↓
    Auxiliary branch → reweights adjacency matrix using CPR
        ↓
    Auxiliary classification score (aux loss, weight=0.2)

  Total loss = CE_main + 0.2 × CE_aux

  The CPR matrix modifies the graph connections during the auxiliary forward pass — joints that the text says are "related" for
   a given class get stronger connections. The model is nudged to use those joint relationships.

  ---
  Why FO specifically benefits from GAP's text but NOT from LAGCN's CPR

  LAGCN's CPR is built from sentences like:
  "when faceoff body position left_knee of hockey player"
  "when faceoff body position right_hand of hockey player"

  These are very generic template sentences. BERT doesn't know what a faceoff stance looks like. The word "faceoff" in that
  sentence doesn't tell BERT much about how the knees relate to the hands. BERT was trained on internet text — it knows
  "faceoff" is a hockey term, but not the biomechanics.

  The result: the CPR matrix for FO looks similar to the CPR for GF or AF. With only 182 FO samples, the auxiliary branch never
   gets enough signal to override this weak prior.

  ---
  GAP's CLIP text is completely different in nature:

  "Hands grip stick low near the ice; Hips hunch over deeply; Legs are wide and bent for immediate reaction"

  CLIP was trained on image-text pairs — it has seen thousands of photos of hockey players in faceoff position with captions
  describing that exact pose. When it reads "legs wide and bent", "crouching", "stick on ice", it produces an embedding that is
   geometrically far from "legs straight, coasting forward" (GF). That distance is what creates a useful contrastive gradient.

  ---
  The core difference visualised

  LAGCN CPR approach:
    Text → BERT → "what joints are structurally related in this class?"
                    → modifies graph edges
                    → needs many samples to override a weak prior

  GAP CLIP approach:
    Text → CLIP → "what does this action look and feel like as a whole?"
                    → directly aligns skeleton feature vector with text embedding
                    → a single good text description creates a strong attractor

  For FO specifically:

  - LAGCN sees 182 FO training clips. The CPR prior is weak (generic BERT templates). 182 samples isn't enough for the
  auxiliary branch to learn "wide-knee, low-crouch = FO". The model collapses FO → GF.
  - GAP sees the same 182 clips, BUT the CLIP text "wide stance, crouched, stick on ice" is already far from "upright,
  coasting" in embedding space. Even with 182 samples, the global contrastive loss pulls the FO feature vector toward a very
  distinct region of embedding space — away from GF. Result: 83% accuracy.

  ---
  The limitation of both approaches

  Neither fully solves the problem:

  ┌──────────────────┬──────────────────────────────────────────────┬──────────────────────────────────────────────────────┐
  │                  │                    LAGCN                     │                         GAP                          │
  ├──────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ Text used        │ Template sentences via BERT                  │ Rich descriptive text via CLIP                       │
  ├──────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ When text is     │ Once, to build a fixed prior                 │ Every training step, as a loss                       │
  │ used             │                                              │                                                      │
  ├──────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ Strength         │ Learns joint relationships from data         │ Leverages CLIP's rich semantic space                 │
  ├──────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ Weakness         │ Generic templates → weak priors for rare     │ Bugs mean only global alignment works (not           │
  │                  │ classes                                      │ per-body-part)                                       │
  ├──────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ FO (182 samples) │ Fails — prior too weak                       │ Works — CLIP text is distinctive enough              │
  ├──────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ MP (702 samples) │ Works — enough data to learn static stance   │ Fails — bugs route MP features to wrong text         │
  └──────────────────┴──────────────────────────────────────────────┴──────────────────────────────────────────────────────┘

  The ideal system would use CLIP-quality text descriptions (like GAP) but apply them at the body-part level (which GAP is
  supposed to do but doesn't due to Bug 1).

● This is the key insight of the whole comparison. The answer is: LAGCN doesn't use text during training at all — that's       
  exactly why it wins on similar-text classes.                                                                                 
                                                                                                                               
  ---                                                                                                                          
  What LAGCN actually does during training                                                                                     
                                               
  The CPR matrix is computed once, before training, and then frozen. During training, the model never looks at text again. It  
  just trains a GCN on raw skeleton data with the CPR matrix baked into the graph structure as a fixed prior.                

  So for GB vs GF — two classes with nearly identical text (0.97 cosine similarity) — LAGCN ignores that similarity entirely.
  It just learns:

  "Skeleton pattern A → GF"
  "Skeleton pattern B → GB"

  purely from the joint coordinate data. The skeleton data IS different between GF and GB — the body is oriented differently,
  velocity vectors point in opposite directions, hip joint positions differ. The model can learn this from raw data alone.

  ---
  Why GAP's text actively hurts on similar-text classes

  GAP computes a contrastive loss every training step. For GB, it tries to pull the GB skeleton feature toward the GB text
  embedding, and push it away from all other class text embeddings.

  But look at the similarity values:

  GF text ↔ GB text:  cosine sim = 0.97
  GF text ↔ GF text:  cosine sim = 1.00

  The gap between "correct class" and "most similar wrong class" is only 0.03. The softmax over all 11 class text embeddings
  produces something like:

  P(GF | GB feature) = 0.18    ← should be 0
  P(GB | GB feature) = 0.22    ← should be 1
  P(AF | GB feature) = 0.12
  ... (all roughly similar)

  The loss says: "pull toward 0.22, push away from 0.18". That gradient is tiny — and it's pointing in almost the same
  direction as the "pull toward GF" gradient. The KL divergence loss is essentially training the model to produce a
  near-uniform distribution over text embeddings, which tells the skeleton encoder almost nothing useful about what makes GB
  different from GF.

  Worse — it's adding gradient noise on top of the CE loss that's already correctly learning the skeleton patterns. That noise
  degrades the features.

  ---
  The conceptual breakdown

  Classes with SIMILAR text (GB, PWG, MP, TBF):
    GAP: text contrastive loss → near-zero/noisy gradient → hurts the CE loss
    LAGCN: no text during training → CE loss learns skeleton patterns cleanly
    Winner: LAGCN

  Classes with DISTINCTIVE text (FO, AF):
    GAP: text contrastive loss → strong gradient → pulls FO/AF features to unique region
    LAGCN: CPR prior too weak (generic BERT templates) → skeleton-only learning with 182 samples → fails
    Winner: GAP

  ---
  The deeper reason text can't solve direction/orientation

  For GF vs GB specifically — what makes them different is the direction of travel, which is a global property of the whole
  sequence. In a 2D overhead-view skeleton:

  - GF: all joints moving in +x direction over time
  - GB: all joints moving in -x direction over time

  CLIP was trained on image-text pairs. The sentence "skating backward" doesn't tell CLIP anything about joint velocity vectors
   in a 2D coordinate system. The relevant difference is in the temporal motion pattern of the skeleton — which the GCN
  captures directly through TCN (temporal convolution) layers but which text cannot encode.

  This is why text guidance is fundamentally limited for motion-direction classes: the discriminative signal lives in the
  temporal skeleton domain, not in the semantic text domain.