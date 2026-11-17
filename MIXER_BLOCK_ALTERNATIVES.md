# Mixer Block Alternatives — Experiment Guide

> **Context:** The MixerSENet architecture isolates spatial-spectral mixing into a single `mixer_block()` function (Cell 6 of the notebook). This document catalogues concrete modifications you can drop into that cell to potentially improve OA/AA/Kappa while preserving the rest of the paper's architecture (stem → mixer blocks × depth → SE → GAP → softmax).

---

## How to Use This Guide

1. Pick an alternative from the table below.
2. Copy its code into **Cell 6** of the notebook, replacing the existing `mixer_block` function.
3. **Keep the function signature unchanged:** `def mixer_block(x, filters: int, kernel_size: int) → tensor`
4. Re-run Cells 7–14. Nothing else needs to change.

---

## Quick Comparison Table

| # | Alternative | Core Idea | Param Δ vs Baseline | FLOP Δ | Expected Benefit |
|---|-------------|-----------|---------------------|--------|------------------|
| A1 | Dilated DepthwiseConv2D | Same receptive field, fewer edge effects | ≈ Same | ≈ Same | Better boundary preservation |
| A2 | Asymmetric Kernels (1×K + K×1) | Factorised spatial mixing | ↓ ~40% | ↓ ~40% | Lighter, less overfitting |
| A3 | Wider Kernel Spread | Kernels 3, 7, 11 instead of 3, 5, 7 | ↑ slight | ↑ slight | Larger context for bigger objects |
| A4 | Attention-Weighted Branch Fusion | Learnable per-branch importance | ↑ negligible | ↑ negligible | Adaptive scale selection |
| A5 | 3D Depthwise Separable | DepthwiseConv3D + pointwise Conv2D | ↑ moderate | ↑ moderate | Explicit spectral-spatial coupling |
| A6 | Hybrid 3D→2D | Conv3D early, Conv2D later | ↑ moderate | ↑ moderate | Spectral in early layers, spatial later |
| A7 | Ghost Module Style | Half filters via Conv, half via cheap DW | ↓ ~50% | ↓ ~50% | Ultra-lightweight |
| A8 | Inverted Bottleneck (MBConv) | Expand → DW → Squeeze | ↑ moderate | ↑ moderate | Richer intermediate representation |
| A9 | Concat + 1×1 Reduce | Concatenate branches then compress | ↑ moderate | ↑ moderate | Preserves per-scale features |
| A10 | ECA (Efficient Channel Attention) in Mixer | Lightweight local channel attention | ↑ negligible | ↑ negligible | Per-block channel recalibration |
| A11 | Spectral Grouping | Group channels, different kernel per group | ≈ Same | ≈ Same | Band-group-specific patterns |
| A12 | ConvNeXt-Style Block | Large kernel DW + LayerNorm + GELU + 1×1 | ≈ Same | ↑ slight | Modern design, strong baselines |
| **B1** | **Center-Conditioned Spatial Gating (CCSG)** | Gate DW branches with center-pixel context | ↑ negligible | ↑ negligible | **Suppresses alien boundary pixels** |
| **B2** | **Center-Spectral Affinity Gating (CSAG)** | Bilateral cosine similarity against center pixel | ↑ negligible | ↑ negligible | **Filters out non-target spectral classes** |
| **B3** | **Central Difference Depthwise Mixing (CDDM)** | Standard DW + Center-Difference DW (CDC) | ≈ Same | ↑ slight | **Gradient/contrast relative to center** |
| **B4** | **Radial-Decayed Center-Prior Gating (RD-CPG)** | Spatial distance prior + center affinity | ↑ negligible | ↑ negligible | **Suppresses distant edge pollution** |
| **B5** | **Center-Prompted Cross-Gating (CPCG)** | Center pixel acts as query to cross-gate DW features | ↑ negligible | ↑ negligible | **Learned center-to-patch modulation** |

---

## Detailed Alternatives

---

### A1 — Dilated (Atrous) Depthwise Convolutions

**Idea:** Replace the large 5×5 and 7×7 kernels with 3×3 kernels using dilation rates of 2 and 3. This achieves the same effective receptive field (5×5 and 7×7) with fewer multiplications and better handling of spatial boundaries.

**Why it may help HSI:** Reduces edge artefacts on small 9×9 patches where large kernels eat into the zero-padded border. Dilated convolutions sample more efficiently across the patch.

```python
def mixer_block(x, filters: int, kernel_size: int):
    residuals = x

    # All 3×3 kernels; dilation expands the receptive field
    pos_emb1 = layers.DepthwiseConv2D(kernel_size=3, padding="same", dilation_rate=1)(x)  # RF = 3×3
    pos_emb2 = layers.DepthwiseConv2D(kernel_size=3, padding="same", dilation_rate=2)(x)  # RF = 5×5
    pos_emb3 = layers.DepthwiseConv2D(kernel_size=3, padding="same", dilation_rate=3)(x)  # RF = 7×7

    x = keras.layers.Add()([residuals, pos_emb1, pos_emb2, pos_emb3])

    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)
    x = layers.Add()([x, residuals])
    return x
```

> [!TIP]
> **Start here.** Minimal change, same param count, often improves edge-pixel accuracy by 1–2%.

---

### A2 — Asymmetric (Factorised) Depthwise Convolutions

**Idea:** Replace each K×K depthwise conv with a pair of 1×K and K×1 convolutions. A 5×5 kernel becomes (1×5 → 5×1), reducing parameters from K² to 2K per channel.

**Why it may help HSI:** Hyperspectral patches are small (9×9). Asymmetric kernels avoid over-parameterisation and can capture directional features (e.g., roads, field edges).

```python
def mixer_block(x, filters: int, kernel_size: int):
    residuals = x

    # Branch 1: 3×3 (kept as-is, already small)
    pos_emb1 = layers.DepthwiseConv2D(kernel_size=3, padding="same")(x)

    # Branch 2: factorised 5×5 → (1×5 → 5×1)
    pos_emb2 = layers.DepthwiseConv2D(kernel_size=(1, 5), padding="same")(x)
    pos_emb2 = layers.DepthwiseConv2D(kernel_size=(5, 1), padding="same")(pos_emb2)

    # Branch 3: factorised 7×7 → (1×7 → 7×1)
    pos_emb3 = layers.DepthwiseConv2D(kernel_size=(1, 7), padding="same")(x)
    pos_emb3 = layers.DepthwiseConv2D(kernel_size=(7, 1), padding="same")(pos_emb3)

    x = keras.layers.Add()([residuals, pos_emb1, pos_emb2, pos_emb3])

    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)
    x = layers.Add()([x, residuals])
    return x
```

---

### A3 — Wider Kernel Spread (3, 7, 11)

**Idea:** Push the multi-scale range wider. The original uses 3, 5, 7 — close together. Using 3, 7, 11 covers a broader spatial context.

**Why it may help HSI:** Datasets like Houston13 have large homogeneous regions (parking lots, highways) that benefit from wider context, while still capturing fine detail at 3×3.

```python
def mixer_block(x, filters: int, kernel_size: int):
    residuals = x

    pos_emb1 = layers.DepthwiseConv2D(kernel_size=3,  padding="same")(x)
    pos_emb2 = layers.DepthwiseConv2D(kernel_size=7,  padding="same")(x)
    pos_emb3 = layers.DepthwiseConv2D(kernel_size=11, padding="same")(x)

    x = keras.layers.Add()([residuals, pos_emb1, pos_emb2, pos_emb3])

    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)
    x = layers.Add()([x, residuals])
    return x
```

> [!WARNING]
> With `WINDOW_SIZE=9`, an 11×11 kernel extends well beyond the patch even with padding. Consider increasing `WINDOW_SIZE` to 13 or 15 if using this variant. Otherwise the outermost kernel is mostly seeing zeros.

---

### A4 — Attention-Weighted Branch Fusion

**Idea:** Instead of a plain sum of the three depthwise branches, learn a soft attention weight for each branch. A small MLP takes the GAP of each branch and outputs 3 scalar weights (softmax-normalised).

**Why it may help HSI:** Different spectral classes may benefit from different spatial scales. Attention lets the network dynamically emphasise the most informative scale per sample.

```python
def mixer_block(x, filters: int, kernel_size: int):
    residuals = x

    pos_emb1 = layers.DepthwiseConv2D(kernel_size=3, padding="same")(x)
    pos_emb2 = layers.DepthwiseConv2D(kernel_size=5, padding="same")(x)
    pos_emb3 = layers.DepthwiseConv2D(kernel_size=7, padding="same")(x)

    # --- Learnable branch-attention weights ---
    # Stack branch GAPs → (batch, 3, C)
    g1 = layers.GlobalAveragePooling2D()(pos_emb1)  # (batch, C)
    g2 = layers.GlobalAveragePooling2D()(pos_emb2)
    g3 = layers.GlobalAveragePooling2D()(pos_emb3)

    g = layers.Concatenate()([g1, g2, g3])                          # (batch, 3C)
    g = layers.Dense(3, activation='softmax', use_bias=False)(g)    # (batch, 3)

    # Reshape weights for broadcasting: (batch, 1, 1, 1) per branch
    w1 = layers.Reshape((1, 1, 1))(layers.Lambda(lambda z: z[:, 0:1])(g))
    w2 = layers.Reshape((1, 1, 1))(layers.Lambda(lambda z: z[:, 1:2])(g))
    w3 = layers.Reshape((1, 1, 1))(layers.Lambda(lambda z: z[:, 2:3])(g))

    weighted = layers.Add()([
        layers.Multiply()([pos_emb1, w1]),
        layers.Multiply()([pos_emb2, w2]),
        layers.Multiply()([pos_emb3, w3]),
    ])
    x = layers.Add()([residuals, weighted])

    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)
    x = layers.Add()([x, residuals])
    return x
```

---

### A5 — 3D Depthwise Separable Convolutions

**Idea:** Reshape the 4-D tensor `(B, H, W, C)` → `(B, H, W, C, 1)`, apply depthwise Conv3D kernels that span both spatial and spectral dimensions, then reshape back and apply pointwise Conv2D.

**Why it may help HSI:** The original paper treats the PCA-reduced bands as flat channels. Conv3D allows the network to explicitly model correlations between *adjacent PCA bands* — these often carry complementary spectral information that channel-independent depthwise 2D misses entirely.

```python
def mixer_block(x, filters: int, kernel_size: int):
    residuals = x

    # Reshape: (B, H, W, C) → (B, H, W, C, 1) to treat channels as a spectral axis
    x_3d = layers.Reshape((*x.shape[1:], 1))(x)  # (B, H, W, C, 1)

    # 3D depthwise convolutions: spatial (3×3) × spectral (3)
    emb1 = layers.Conv3D(1, kernel_size=(3, 3, 3), padding="same", groups=1)(x_3d)
    emb2 = layers.Conv3D(1, kernel_size=(5, 5, 3), padding="same", groups=1)(x_3d)
    emb3 = layers.Conv3D(1, kernel_size=(7, 7, 3), padding="same", groups=1)(x_3d)

    x_3d = keras.layers.Add()([x_3d, emb1, emb2, emb3])

    # Reshape back: (B, H, W, C, 1) → (B, H, W, C)
    x = layers.Reshape(x.shape[1:])(x_3d)

    # Pointwise channel mixing (same as original)
    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)
    x = layers.Add()([x, residuals])
    return x
```

> [!IMPORTANT]
> This is the modification closest to what your task description mentioned ("3× Depthwise Conv3D + pointwise Conv2D"). The reshape trick avoids changing the MixerSENet assembly at all.

> [!NOTE]
> The `groups=1` Conv3D is not truly depthwise in the 3D sense (Keras doesn't natively support DepthwiseConv3D). For a true depthwise 3D, you'd need `groups=C`, which requires `filters` to be a multiple of `C`. See A5b below for a workaround.

#### A5b — True Depthwise Conv3D (via grouped convolutions)

```python
def mixer_block(x, filters: int, kernel_size: int):
    residuals = x
    C = int(x.shape[-1])

    x_3d = layers.Reshape((*x.shape[1:3], 1, C))(x)  # (B, H, W, 1, C)

    # Grouped Conv3D with groups=C acts as depthwise
    emb1 = layers.Conv3D(C, kernel_size=(3, 3, 1), padding="same", groups=C)(x_3d)
    emb2 = layers.Conv3D(C, kernel_size=(5, 5, 1), padding="same", groups=C)(x_3d)
    emb3 = layers.Conv3D(C, kernel_size=(7, 7, 1), padding="same", groups=C)(x_3d)

    x_3d = keras.layers.Add()([x_3d, emb1, emb2, emb3])

    x = layers.Reshape(x.shape[1:])(x_3d)

    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)
    x = layers.Add()([x, residuals])
    return x
```

---

### A6 — Hybrid 3D→2D (Spectral Conv3D + Spatial DepthwiseConv2D)

**Idea:** First apply a small Conv3D to mix adjacent spectral bands, then fall back to spatial depthwise Conv2D for multi-scale spatial mixing.

**Why it may help HSI:** Decouples spectral mixing (needs small kernels along band axis) from spatial mixing (needs multi-scale). Each operation is specialised.

```python
def mixer_block(x, filters: int, kernel_size: int):
    residuals = x
    C = int(x.shape[-1])

    # --- Spectral mixing via Conv3D ---
    x_3d = layers.Reshape((*x.shape[1:], 1))(x)          # (B, H, W, C, 1)
    x_3d = layers.Conv3D(1, kernel_size=(1, 1, 3), padding="same")(x_3d)  # spectral-only
    x_spec = layers.Reshape(x.shape[1:])(x_3d)            # (B, H, W, C)

    # --- Multi-scale spatial mixing via DepthwiseConv2D ---
    pos_emb1 = layers.DepthwiseConv2D(kernel_size=3, padding="same")(x_spec)
    pos_emb2 = layers.DepthwiseConv2D(kernel_size=5, padding="same")(x_spec)
    pos_emb3 = layers.DepthwiseConv2D(kernel_size=7, padding="same")(x_spec)

    x = keras.layers.Add()([residuals, pos_emb1, pos_emb2, pos_emb3])

    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)
    x = layers.Add()([x, residuals])
    return x
```

---

### A7 — Ghost Module Style

**Idea:** (From GhostNet) Generate half the output filters with a standard Conv, then produce the other half via cheap depthwise transforms of the first half. Cuts params and FLOPs by ~50%.

**Why it may help HSI:** When you're training on very few samples (Houston13 has ~650 train samples), a lighter block reduces overfitting risk.

```python
def mixer_block(x, filters: int, kernel_size: int):
    residuals = x
    half = filters // 2

    # Primary features: cheap 1×1 conv for half the filters
    primary = layers.Conv2D(half, kernel_size=1)(x)
    primary = activation_block(primary)

    # Ghost features: cheap depthwise transforms of the primary
    ghost1 = layers.DepthwiseConv2D(kernel_size=3, padding="same")(primary)
    ghost2 = layers.DepthwiseConv2D(kernel_size=5, padding="same")(primary)

    # Concatenate primary + ghosts, then project back to `filters`
    x = layers.Concatenate()([primary, ghost1, ghost2])  # half + half + half = 1.5× filters
    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)
    x = layers.Add()([x, residuals])
    return x
```

---

### A8 — Inverted Bottleneck (MBConv-Style)

**Idea:** (From MobileNetV2 / EfficientNet) Expand channel dimension by a factor, apply depthwise conv in the expanded space, then compress back. Richer intermediate representations.

```python
def mixer_block(x, filters: int, kernel_size: int):
    residuals = x
    expand_ratio = 4

    # Expand
    x = layers.Conv2D(filters * expand_ratio, kernel_size=1)(x)
    x = activation_block(x)

    # Multi-scale depthwise in expanded space
    pos_emb1 = layers.DepthwiseConv2D(kernel_size=3, padding="same")(x)
    pos_emb2 = layers.DepthwiseConv2D(kernel_size=5, padding="same")(x)
    pos_emb3 = layers.DepthwiseConv2D(kernel_size=7, padding="same")(x)
    x = keras.layers.Add()([x, pos_emb1, pos_emb2, pos_emb3])

    # Squeeze back
    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = layers.BatchNormalization()(x)  # No activation before residual (linear bottleneck)

    x = layers.Add()([x, residuals])
    return x
```

> [!NOTE]
> `expand_ratio=4` means 4× more channels in the depthwise step. This increases FLOPs but gives the depthwise convolutions a much richer feature space to work with. Try `expand_ratio=2` first for a lighter version.

---

### A9 — Concatenate Branches + 1×1 Reduce

**Idea:** Instead of summing the three depthwise branches (which forces them to share the same feature space), concatenate them and let a 1×1 conv learn the optimal combination.

**Why it may help HSI:** Addition destroys per-scale information. Concatenation preserves it, and the 1×1 conv can learn which scale matters for which channel.

```python
def mixer_block(x, filters: int, kernel_size: int):
    residuals = x

    pos_emb1 = layers.DepthwiseConv2D(kernel_size=3, padding="same")(x)
    pos_emb2 = layers.DepthwiseConv2D(kernel_size=5, padding="same")(x)
    pos_emb3 = layers.DepthwiseConv2D(kernel_size=7, padding="same")(x)

    # Concatenate instead of Add: (B, H, W, 4C)
    x = layers.Concatenate()([residuals, pos_emb1, pos_emb2, pos_emb3])

    # Project back to original channel count
    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)
    x = layers.Add()([x, residuals])
    return x
```

> [!TIP]
> This is a strong baseline modification. The 1×1 conv after concatenation has `4C × filters` parameters (vs `C × filters` in the original), but it can learn much richer scale-specific combinations.

---

### A10 — Per-Block ECA (Efficient Channel Attention)

**Idea:** Add a lightweight 1-D channel attention *inside* the mixer block (after the depthwise sum, before the pointwise conv). ECA uses a single 1-D conv on the GAP vector — no Dense layers, no reduction ratio.

**Why it may help HSI:** The global SE block at the end only recalibrates once. Adding per-block attention lets each mixer layer suppress noisy bands early.

```python
def mixer_block(x, filters: int, kernel_size: int):
    residuals = x

    pos_emb1 = layers.DepthwiseConv2D(kernel_size=3, padding="same")(x)
    pos_emb2 = layers.DepthwiseConv2D(kernel_size=5, padding="same")(x)
    pos_emb3 = layers.DepthwiseConv2D(kernel_size=7, padding="same")(x)

    x = keras.layers.Add()([residuals, pos_emb1, pos_emb2, pos_emb3])

    # --- ECA: Efficient Channel Attention ---
    gap = layers.GlobalAveragePooling2D()(x)                       # (B, C)
    gap = layers.Reshape((int(x.shape[-1]), 1))(gap)               # (B, C, 1)
    eca = layers.Conv1D(1, kernel_size=5, padding="same")(gap)     # (B, C, 1)
    eca = layers.Activation("sigmoid")(eca)
    eca = layers.Reshape((1, 1, int(x.shape[-1])))(eca)            # (B, 1, 1, C)
    x = layers.Multiply()([x, eca])
    # --- End ECA ---

    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)
    x = layers.Add()([x, residuals])
    return x
```

---

### A11 — Spectral Grouping (Group-Specific Kernels)

**Idea:** Divide the `C` channels into groups (e.g., 4 groups of 16 for C=64). Apply a different kernel size to each group's depthwise conv. Different spectral band groups may have different spatial correlation scales.

```python
def mixer_block(x, filters: int, kernel_size: int):
    residuals = x
    C = int(x.shape[-1])
    G = 4  # number of groups
    per_group = C // G

    groups = []
    for g, ks in enumerate([3, 5, 7, 9]):
        x_g = layers.Lambda(lambda t, s=g: t[..., s*per_group:(s+1)*per_group])(x)
        x_g = layers.DepthwiseConv2D(kernel_size=ks, padding="same")(x_g)
        groups.append(x_g)

    x = layers.Concatenate()(groups)  # (B, H, W, C)
    x = layers.Add()([x, residuals])

    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)
    x = layers.Add()([x, residuals])
    return x
```

---

### A12 — ConvNeXt-Style Block

**Idea:** Borrow the modern ConvNeXt design: single large-kernel depthwise conv → LayerNorm → 1×1 expand → GELU → 1×1 compress. Proven strong on many vision benchmarks.

```python
def mixer_block(x, filters: int, kernel_size: int):
    residuals = x

    # Single large-kernel depthwise conv (replaces 3 parallel branches)
    x = layers.DepthwiseConv2D(kernel_size=7, padding="same")(x)
    x = layers.LayerNormalization()(x)

    # Inverted bottleneck: expand → GELU → compress
    x = layers.Conv2D(filters * 4, kernel_size=1)(x)
    x = layers.Activation("gelu")(x)
    x = layers.Conv2D(filters, kernel_size=1)(x)

    x = layers.Add()([x, residuals])
    return x
```

> [!NOTE]
> This drops `activation_block()` (GELU + BN) in favour of LayerNorm + inline GELU, matching ConvNeXt conventions. The rest of the architecture (stem, SE, head) stays unchanged.

---

## Group B — Center-Conditioned & Anchor-Guided Mixers

> **Theoretical Motivation:**  
> In patch-based HSI classification, each $W \times W$ patch (here $9 \times 9$) is labeled **strictly by the class of its center pixel** $(H//2, W//2)$.  
> Standard depthwise convolutions process all spatial pixels uniformly. However, at class boundaries or in complex urban scenes (like Houston13), neighboring pixels inside the patch frequently belong to *different classes*. Large kernels ($5 \times 5, 7 \times 7$) can inadvertently pool alien spectral signatures and contaminate the center target's features.  
> 
> **Center-Conditioned blocks** treat the center pixel as an **anchor**: they evaluate how consistent surrounding pixels are with the center, or use the center representation to gate the multi-scale spatial features. All blocks below preserve the exact `mixer_block(x, filters, kernel_size)` signature.

---

### B1 — Center-Conditioned Spatial Gating (CCSG)

**Idea:** Slice out the center pixel vector $(1, 1, C)$, broadcast it across the spatial grid, concatenate it with the spatial feature map, and pass it through a lightweight projection to generate a soft spatial-attentive gate $G \in [0, 1]^{(B, H, W, 1)}$.

**Why it helps HSI:** The gate dynamically down-weights spatial locations whose features conflict with or are irrelevant to the center pixel, preventing boundary pixel pollution before the pointwise conv.

```python
def mixer_block(x, filters: int, kernel_size: int):
    residuals = x
    H, W = x.shape[1], x.shape[2]
    c_h, c_w = H // 2, W // 2

    # --- 1. Multi-scale depthwise convolution (same as baseline) ---
    pos_emb1 = layers.DepthwiseConv2D(kernel_size=3, padding="same")(x)
    pos_emb2 = layers.DepthwiseConv2D(kernel_size=5, padding="same")(x)
    pos_emb3 = layers.DepthwiseConv2D(kernel_size=7, padding="same")(x)
    spatial_feats = keras.layers.Add()([residuals, pos_emb1, pos_emb2, pos_emb3])

    # --- 2. Center-Conditioned Spatial Gating (CCSG) ---
    # Extract the center pixel: shape (B, 1, 1, C)
    center_token = layers.Lambda(lambda t: t[:, c_h:c_h+1, c_w:c_w+1, :])(x)
    # Broadcast center token to (B, H, W, C)
    center_map = layers.Lambda(lambda t, h=H, w=W: tf.tile(t, [1, h, w, 1]))(center_token)

    # Condition spatial positions on center context
    cond = layers.Concatenate()([spatial_feats, center_map])           # (B, H, W, 2C)
    gate = layers.Conv2D(filters // 4, kernel_size=1, activation="relu")(cond)
    gate = layers.Conv2D(1, kernel_size=1, activation="sigmoid")(gate) # (B, H, W, 1)

    # Modulate multi-scale spatial features with center gate
    gated_spatial = layers.Multiply()([spatial_feats, gate])

    # --- 3. Pointwise channel mixing (same as baseline) ---
    x = layers.Conv2D(filters, kernel_size=1)(gated_spatial)
    x = activation_block(x)
    x = layers.Add()([x, residuals])
    return x
```

---

### B2 — Center-Spectral Affinity Gating (CSAG / Bilateral Mixer)

**Idea:** Inspired by bilateral filtering in image processing. Compute the normalized dot-product (cosine affinity) between every spatial pixel's spectral feature and the center pixel's spectral feature.

**Why it helps HSI:** Hyperspectral classification relies heavily on spectral curve similarity. Pixels within the $9 \times 9$ window that have similar spectral signatures to the center receive high weights (~1.0); dissimilar pixels (e.g., grass next to concrete) receive low weights (~0.0).

```python
def mixer_block(x, filters: int, kernel_size: int):
    residuals = x
    H, W = x.shape[1], x.shape[2]
    c_h, c_w = H // 2, W // 2

    pos_emb1 = layers.DepthwiseConv2D(kernel_size=3, padding="same")(x)
    pos_emb2 = layers.DepthwiseConv2D(kernel_size=5, padding="same")(x)
    pos_emb3 = layers.DepthwiseConv2D(kernel_size=7, padding="same")(x)
    spatial_feats = keras.layers.Add()([residuals, pos_emb1, pos_emb2, pos_emb3])

    # --- Center-Spectral Affinity (Bilateral Gating) ---
    center_token = layers.Lambda(lambda t: t[:, c_h:c_h+1, c_w:c_w+1, :])(x)  # (B, 1, 1, C)

    # Normalize vectors to compute cosine similarity across channels
    def cosine_affinity(tensors):
        feat, center = tensors
        feat_norm = tf.nn.l2_normalize(feat, axis=-1)
        center_norm = tf.nn.l2_normalize(center, axis=-1)
        sim = tf.reduce_sum(feat_norm * center_norm, axis=-1, keepdims=True)  # (B, H, W, 1)
        return tf.nn.sigmoid(sim * 5.0)  # Temperature scaling and mapped to [0, 1]

    affinity_gate = layers.Lambda(cosine_affinity)([x, center_token])

    # Modulate spatial features with spectral affinity
    gated_spatial = layers.Multiply()([spatial_feats, affinity_gate])

    x = layers.Conv2D(filters, kernel_size=1)(gated_spatial)
    x = activation_block(x)
    x = layers.Add()([x, residuals])
    return x
```

---

### B3 — Central Difference Depthwise Mixing (CDDM)

**Idea:** Based on *Central Difference Convolution (CDC)*. Decomposes the spatial mixing into two distinct complementary streams:
1. Standard depthwise convolution (capturing absolute intensity / contextual semantics)
2. Central difference depthwise convolution on $(x - x_{center})$ (capturing fine-grained local gradient / contrast relative to the target label)

```python
def mixer_block(x, filters: int, kernel_size: int):
    residuals = x
    H, W = x.shape[1], x.shape[2]
    c_h, c_w = H // 2, W // 2

    # --- 1. Standard Multi-scale Depthwise (Intensity context) ---
    std_1 = layers.DepthwiseConv2D(kernel_size=3, padding="same")(x)
    std_2 = layers.DepthwiseConv2D(kernel_size=5, padding="same")(x)
    std_stream = keras.layers.Add()([std_1, std_2])

    # --- 2. Central Difference Stream (Gradient/contrast relative to center) ---
    center_token = layers.Lambda(lambda t: t[:, c_h:c_h+1, c_w:c_w+1, :])(x)
    # Relative difference map: x_diff = x - x_center
    x_diff = layers.Lambda(lambda tensors: tensors[0] - tensors[1])([x, center_token])
    diff_1 = layers.DepthwiseConv2D(kernel_size=3, padding="same")(x_diff)
    diff_2 = layers.DepthwiseConv2D(kernel_size=5, padding="same")(x_diff)
    diff_stream = keras.layers.Add()([diff_1, diff_2])

    # Combine regular features + central-difference features
    mixed_spatial = keras.layers.Add()([residuals, std_stream, diff_stream])

    # --- 3. Pointwise projection ---
    x = layers.Conv2D(filters, kernel_size=1)(mixed_spatial)
    x = activation_block(x)
    x = layers.Add()([x, residuals])
    return x
```

---

### B4 — Radial-Decayed Center-Prior Gating (RD-CPG)

**Idea:** Spatial distance prior + center affinity. In remote sensing, spatial autocorrelation (Tobler's First Law of Geography) implies pixels closer to the center are much more likely to share the center's class than pixels near the edge of the patch. RD-CPG combines a fixed radial Gaussian spatial attenuation prior with a learned center gate.

```python
def mixer_block(x, filters: int, kernel_size: int):
    residuals = x
    H, W = x.shape[1], x.shape[2]
    c_h, c_w = H // 2, W // 2

    pos_emb1 = layers.DepthwiseConv2D(kernel_size=3, padding="same")(x)
    pos_emb2 = layers.DepthwiseConv2D(kernel_size=5, padding="same")(x)
    pos_emb3 = layers.DepthwiseConv2D(kernel_size=7, padding="same")(x)
    spatial_feats = keras.layers.Add()([residuals, pos_emb1, pos_emb2, pos_emb3])

    # Construct static 2D Gaussian radial decay mask centered at (c_h, c_w)
    def make_radial_decay_mask(h, w, sigma=2.5):
        ys, xs = np.ogrid[:h, :w]
        dist_sq = (ys - c_h)**2 + (xs - c_w)**2
        decay = np.exp(-dist_sq / (2.0 * sigma**2)).astype(np.float32)
        return decay[np.newaxis, :, :, np.newaxis] # (1, H, W, 1)

    decay_prior = tf.constant(make_radial_decay_mask(H, W), dtype=tf.float32)

    # Center-token interaction
    center_token = layers.Lambda(lambda t: t[:, c_h:c_h+1, c_w:c_w+1, :])(x)
    center_map = layers.Lambda(lambda t, h=H, w=W: tf.tile(t, [1, h, w, 1]))(center_token)
    
    # Dynamic gate modulated by static geometric distance prior
    diff = layers.Lambda(lambda t: tf.abs(t[0] - t[1]))([x, center_map])
    dynamic_gate = layers.Conv2D(1, kernel_size=1, activation="sigmoid")(diff)
    final_gate = layers.Multiply()([dynamic_gate, decay_prior])

    gated_spatial = layers.Multiply()([spatial_feats, final_gate])

    x = layers.Conv2D(filters, kernel_size=1)(gated_spatial)
    x = activation_block(x)
    x = layers.Add()([x, residuals])
    return x
```

---

### B5 — Center-Prompted Cross-Gating (CPCG)

**Idea:** The center pixel acts as an explicit "query / prompt" vector, while the multi-scale depthwise features act as "keys". A single dot-product generates a channel-wise or spatial gating map.

**Why it helps HSI:** Instead of computing heavy $O((HW)^2)$ full self-attention across all patch tokens, center cross-gating costs only $O(HW \cdot C)$. It queries which spatial tokens are relevant to decoding the center token's class.

```python
def mixer_block(x, filters: int, kernel_size: int):
    residuals = x
    H, W, C = x.shape[1], x.shape[2], int(x.shape[-1])
    c_h, c_w = H // 2, W // 2

    # Multi-scale depthwise branches
    pos_emb1 = layers.DepthwiseConv2D(kernel_size=3, padding="same")(x)
    pos_emb2 = layers.DepthwiseConv2D(kernel_size=5, padding="same")(x)
    pos_emb3 = layers.DepthwiseConv2D(kernel_size=7, padding="same")(x)
    spatial_feats = keras.layers.Add()([residuals, pos_emb1, pos_emb2, pos_emb3])

    # Center token as Query: (B, 1, C)
    center_q = layers.Lambda(lambda t: tf.reshape(t[:, c_h, c_w, :], [-1, 1, C]))(x)
    center_q = layers.Dense(C // 2)(center_q)

    # Spatial features as Keys: (B, H*W, C//2)
    spatial_flat = layers.Reshape((H * W, C))(spatial_feats)
    spatial_k = layers.Dense(C // 2)(spatial_flat)

    # Dot-product cross-attention: (B, 1, C//2) @ (B, C//2, H*W) -> (B, 1, H*W)
    attn = layers.Lambda(lambda tensors: tf.matmul(tensors[0], tensors[1], transpose_b=True))([center_q, spatial_k])
    attn = layers.Lambda(lambda a: tf.nn.softmax(a / ((C // 2) ** 0.5), axis=-1))(attn)
    attn_map = layers.Reshape((H, W, 1))(attn)

    # Modulate spatial features with normalized center-prompted attention
    # Scaled by (H * W) so mean weight is ~1.0
    attn_map_scaled = layers.Lambda(lambda a, scale=H*W: a * scale)(attn_map)
    gated_spatial = layers.Multiply()([spatial_feats, attn_map_scaled])

    x = layers.Conv2D(filters, kernel_size=1)(gated_spatial)
    x = activation_block(x)
    x = layers.Add()([x, residuals])
    return x
```

---

## Recommended Experiment Order

For a major project, I'd suggest this progression — each builds understanding:

| Priority | Variant | Rationale |
|----------|---------|-----------|
| 🥇 1st | **Baseline** (original) | Establish your reference OA/AA/Kappa numbers |
| 🥈 2nd | **A1** (Dilated) | Minimal change, isolates the effect of dilation |
| 🥉 3rd | **A9** (Concat + Reduce) | Tests whether preserving per-scale features matters |
| 4th | **A4** (Attention Fusion) | Tests whether adaptive scale selection helps |
| 5th | **A10** (ECA in Mixer) | Tests whether per-block channel attention adds value on top of the final SE |
| 6th | **A5 / A6** (3D variants) | Tests explicit spectral mixing — key hypothesis for HSI |
| 7th | **A8** (MBConv) | Tests whether richer intermediate space helps |
| 8th | **A12** (ConvNeXt) | Completely different design philosophy — strong reference point |

---

## Experimental Tips

1. **Run each variant 3–5 times** with different random seeds and report mean ± std for OA/AA/Kappa. Single-run comparisons on small datasets like Houston13 are unreliable.

2. **Control the comparison:** Change ONLY the mixer block. Same `DEPTH=5`, `FILTERS=64`, `WINDOW_SIZE=9`, `EPOCHS=100`, `BATCH_SIZE=64`, `LEARNING_RATE=5e-4`.

3. **Track FLOPs:** Run `net_flops(model)` for each variant and include it in your results table. A variant that improves Kappa by 0.5% but doubles FLOPs may not be worth it for "lightweight."

4. **Ablation matters:** If A9 (Concat) beats the baseline, also test 2 branches instead of 3, or 4 branches — this isolates whether it's the *number* of scales or the *fusion method* that helped.

5. **Combine winners:** If both A1 (Dilated) and A10 (ECA) individually improve results, try them together. The modifications are orthogonal.

---

## References

- **GhostNet** (A7): Han et al., "GhostNet: More Features from Cheap Operations," CVPR 2020
- **MobileNetV2** (A8): Sandler et al., "Inverted Residuals and Linear Bottlenecks," CVPR 2018
- **ECA-Net** (A10): Wang et al., "ECA-Net: Efficient Channel Attention," CVPR 2020
- **ConvNeXt** (A12): Liu et al., "A ConvNet for the 2020s," CVPR 2022
- **SS-ConvNeXt for HSI**: Spatial-Spectral ConvNeXt adaptations for hyperspectral classification, 2024
- **HDCDF**: Hybrid Depthwise Separable Conv + Dual-Branch Feature Fusion for HSI, 2025
- **LRCNet**: Lightweight Receptive Control with 3D Depthwise Separable Conv for HSI, 2024
