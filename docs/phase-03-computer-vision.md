# Phase 3: Computer Vision

**Milestone:** M2 — "I Have Data" (second half)
**Prerequisites:** [Phase 2: Data Pipeline](phase-02-data-pipeline.md) complete — demonstrations collected, Dataset working
**Deliverables:** Vision encoder integrated, feature extraction verified, understanding of spatial vs. global features

---

## Overview

The robot needs to "see." Raw pixels (128x128x3 = 49,152 numbers) are too high-dimensional and too unstructured for the policy network to work with directly. A vision encoder compresses each image into a compact, meaningful representation — a set of feature vectors that capture what objects are present and where they are.

This phase teaches you the computer vision foundations underlying EchoTask: how Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) work, why we use pretrained encoders instead of training from scratch, and the critical difference between spatial (patch-level) features and global features.

---

## Concepts

### Why Not Use Raw Pixels?

A 128x128 RGB image has 49,152 values. Most of these are redundant — the pixel at position (64, 65) is almost identical to its neighbor at (64, 66). Worse, raw pixels don't capture semantic meaning: the number "0.8" in the red channel at position (50, 30) tells you nothing about whether there's a red block there.

A vision encoder transforms raw pixels into a **feature space** where:

- Semantically similar things have similar feature vectors (all "red blocks" cluster together)
- Spatial layout is preserved (features know where objects are, not just what they are)
- Dimensionality is reduced (from 49,152 values to, say, 256 values per spatial location)

### Convolutional Neural Networks (CNNs) — The Foundation

CNNs are the original deep learning approach to vision. Understanding them helps you understand what ViTs improved upon.

**How a convolution works:**

1. A small filter (e.g., 3x3 pixels) slides across the image
2. At each position, it computes a weighted sum of the overlapping pixels
3. The result is a new image (feature map) where each pixel represents "how much this local region matches the filter pattern"
4. Different filters detect different patterns: edges, corners, textures, colors

**Hierarchical feature learning:**

- **Early layers** detect low-level features: edges, color gradients, simple textures
- **Middle layers** combine these into parts: corners become "rectangular shape," color gradients become "red surface"
- **Deep layers** compose parts into objects: "rectangular red surface" becomes "red block"

**The key CNN limitation for EchoTask:** Standard CNNs produce a single global feature vector (e.g., after global average pooling). This tells you **what** is in the image but loses **where** it is. EchoTask needs spatial information — the policy must know the red block is at the left side of the table, not just that a red block exists somewhere.

### Vision Transformers (ViTs) — The Modern Approach

Vision Transformers adapt the Transformer architecture (originally designed for text) to images. Here's how:

**Step 1 — Patch embedding:** The image is divided into non-overlapping patches (e.g., 16x16 pixels each). A 128x128 image becomes 64 patches (8x8 grid). Each patch is flattened and linearly projected into a feature vector (e.g., 768 dimensions).

**Step 2 — Positional encoding:** Each patch embedding gets a positional encoding added, so the model knows where in the image each patch came from. Without this, the model couldn't distinguish "red block on the left" from "red block on the right."

**Step 3 — Transformer layers:** The sequence of patch embeddings passes through standard Transformer encoder layers (self-attention + feed-forward). Self-attention lets every patch attend to every other patch — the model learns relationships between distant image regions.

**Step 4 — Output:** The Transformer outputs one feature vector per patch. This is the critical difference from CNNs: **you get a grid of spatial features, not a single global vector.** Each feature vector describes what's happening at that spatial location in the image.

For EchoTask, this means:
- Patch at position (2, 3) might produce a feature that encodes "there's a red block here"
- Patch at position (5, 6) might encode "there's a blue cylinder here"
- The policy network can attend to specific spatial locations to decide where to move

### Pretrained Encoders: Standing on Giants' Shoulders

Training a vision encoder from scratch on your ~500 demonstrations would be hopeless — you don't have nearly enough visual data to learn what "red block" or "cylinder" looks like from scratch.

Instead, you use a **pretrained encoder** — a vision model trained on millions of diverse images. These models have already learned:

- Low-level features: edges, textures, colors
- Mid-level features: shapes, surfaces, object parts
- High-level features: object categories, spatial relationships

You download the pretrained weights and use the encoder as a fixed feature extractor. The key term is **frozen** — you don't update the encoder's weights during EchoTask training. Only the downstream layers (projection, temporal transformer, policy) are trained.

**Why freeze the encoder?**

1. **Data efficiency:** You have ~100K training images. The encoder was pretrained on millions. Fine-tuning on your small dataset would overfit and destroy the general features.
2. **Compute savings:** A frozen encoder's forward pass doesn't require gradient computation. This roughly halves memory usage and speeds up training.
3. **Stability:** The features don't change during training, so the downstream networks train on a stable input distribution.

### SigLIP vs. DINOv2: The Two Candidates

EchoTask considers two pretrained vision encoders. Understanding their differences helps you choose.

#### SigLIP (Sigmoid Loss for Language-Image Pre-Training)

- **Training method:** Trained to align images with text descriptions. Given an (image, caption) pair, SigLIP learns to produce similar feature vectors for the image and its description.
- **What it learns:** Semantic features — "this image contains a red block on a table." Strong at recognizing objects and their attributes.
- **Architecture:** ViT-B/16 (Base size, 16x16 patches). Outputs 64 patch features of dimension 768 for a 128x128 input.
- **Strengths for EchoTask:** Excellent at identifying object types and colors, which is what the demonstration encoder needs.
- **Weaknesses:** May lose fine-grained spatial detail because it was trained for image-level understanding, not pixel-level.

#### DINOv2 (Self-Distillation with No Labels v2)

- **Training method:** Self-supervised learning — trained without any text labels. The model learns by comparing different augmented views of the same image, forcing it to discover visual structure on its own.
- **What it learns:** Fine-grained visual features — texture, shape, spatial layout. Particularly strong at spatial correspondence (knowing that "this patch in image A corresponds to that patch in image B").
- **Architecture:** ViT-B/14 (Base size, 14x14 patches). Outputs 81 patch features of dimension 768 for a 126x126 input (or interpolated for 128x128).
- **Strengths for EchoTask:** Superior spatial features — excellent at localizing objects precisely. The scene perceiver benefits from knowing exactly where objects are.
- **Weaknesses:** Less semantic — might not differentiate "red block" from "red cylinder" as strongly as SigLIP.

#### Which Should You Use?

Start with **SigLIP** — its semantic features are a better match for the demonstration encoder, which needs to understand task intent (which object was manipulated). If spatial accuracy becomes a bottleneck for the scene perceiver, experiment with DINOv2.

A hybrid approach (SigLIP for the demonstration encoder, DINOv2 for the scene perceiver) is also viable but adds complexity.

### Spatial Features vs. Global Features

This distinction is fundamental to EchoTask's architecture:

**Global features (CLS token):** A single vector summarizing the entire image. Useful for classification ("this image contains a red block") but not for spatial reasoning ("the red block is 10cm to the left of the blue one").

**Spatial features (patch tokens):** A grid of vectors, one per image patch. Each vector describes a local region. The policy can attend to specific locations — "look at the patch where the red block is, now look at the patch where the blue block is, now plan a trajectory."

EchoTask uses **both**:

- **Demonstration encoder:** Takes the global CLS token (or mean-pooled patch features) from each demo frame, since it needs to understand the overall task, not spatial details of the demo scene
- **Scene perceiver:** Takes the full grid of patch features from the current scene image, since the policy needs precise spatial information to generate actions

---

## Implementation Guide

### Step 1: Load the Pretrained Encoder

Using Hugging Face `transformers`:

```python
from transformers import AutoModel, AutoProcessor

# SigLIP
encoder = AutoModel.from_pretrained("google/siglip-base-patch16-224")
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

# DINOv2
encoder = AutoModel.from_pretrained("facebook/dinov2-base")
processor = AutoProcessor.from_pretrained("facebook/dinov2-base")
```

Freeze all parameters:

```python
for param in encoder.parameters():
    param.requires_grad = False
encoder.eval()
```

### Step 2: Understand the Output Structure

Run a sample image through the encoder and inspect the output:

```python
import torch

# Create a dummy 128x128 image (or load a real one from your dataset)
dummy_image = torch.randn(1, 3, 128, 128)

with torch.no_grad():
    outputs = encoder(pixel_values=dummy_image)

# For ViT-based models:
# outputs.last_hidden_state shape: (batch, num_patches + 1, hidden_dim)
# The +1 is the CLS token (first position)

cls_token = outputs.last_hidden_state[:, 0, :]        # (1, 768) — global feature
patch_tokens = outputs.last_hidden_state[:, 1:, :]    # (1, 64, 768) — spatial features
```

Verify:
- CLS token shape: (batch_size, 768)
- Patch tokens shape: (batch_size, num_patches, 768)
- For SigLIP with 16x16 patches on 128x128 input: num_patches = (128/16)^2 = 64
- For DINOv2 with 14x14 patches on 128x128 input: num_patches ≈ 81 (after interpolation)

### Step 3: Build the Feature Extraction Module

Create a wrapper in `backend/models/vision_encoder.py` that:

1. Loads the pretrained encoder and freezes it
2. Handles input preprocessing (resize, normalize to encoder's expected format)
3. Extracts both CLS token and patch features
4. Projects features to the model's internal dimension (256 or 512) via a learned linear layer

```python
class VisionFeatureExtractor(nn.Module):
    def __init__(self, encoder_name="siglip", project_dim=256):
        super().__init__()
        self.encoder = load_pretrained_encoder(encoder_name)
        freeze(self.encoder)

        hidden_dim = 768  # ViT-Base output dimension
        self.global_projection = nn.Linear(hidden_dim, project_dim)
        self.spatial_projection = nn.Linear(hidden_dim, project_dim)

    def forward(self, images):
        with torch.no_grad():
            features = self.encoder(images).last_hidden_state

        cls_feature = features[:, 0, :]        # Global
        patch_features = features[:, 1:, :]    # Spatial

        global_feat = self.global_projection(cls_feature)
        spatial_feat = self.spatial_projection(patch_features)

        return global_feat, spatial_feat
```

The projection layers (`global_projection`, `spatial_projection`) are **trainable** — they learn to map the generic pretrained features into the task-specific embedding space.

### Step 4: Resolution Handling

The pretrained encoders expect specific input sizes (e.g., 224x224 for SigLIP). Your training images are 128x128. Options:

1. **Resize to encoder's expected size** — simplest, slight quality loss from upscaling
2. **Interpolate position embeddings** — modify the encoder's positional embeddings to handle 128x128 natively. More complex but avoids upscaling artifacts

Start with option 1 (resize images to 224x224 before feeding to the encoder). If you observe quality issues, try option 2.

### Step 5: Verify Feature Quality

Run a sanity check: extract features from 20 diverse scene images and visualize:

1. **Cosine similarity matrix:** Compute pairwise cosine similarities between global features. Images with similar scenes (same object types) should have higher similarity.
2. **Patch feature visualization:** For a single image, reshape the patch features into a spatial grid and visualize the first few principal components as a heatmap. Object locations should be clearly distinguishable.
3. **Feature norms:** Check that feature vectors have reasonable magnitudes (not near-zero or exploding).

---

## Learning Checkpoints

**Conceptual checks:**

- [ ] What is the difference between a CNN's output and a ViT's output, and why does it matter for EchoTask?
- [ ] Why do we freeze the pretrained encoder instead of fine-tuning it?
- [ ] What does "patch feature" mean spatially — what part of the image does patch (3, 4) correspond to?
- [ ] Why might SigLIP features be better for the demonstration encoder and DINOv2 features for the scene perceiver?
- [ ] What is the CLS token and how does it differ from the mean of all patch tokens?

**Practical milestones:**

- [ ] Pretrained encoder loaded and frozen (verify zero gradients)
- [ ] Forward pass produces correct output shapes (CLS + patches)
- [ ] Projection layers map from 768-dim to 256-dim (trainable)
- [ ] Feature extraction works on a batch of 128x128 images from the dataset
- [ ] Cosine similarity sanity check shows meaningful feature structure
- [ ] Patch feature heatmap shows object localization

---

## Key Files

| File | Purpose |
|------|---------|
| `backend/models/vision_encoder.py` | Pretrained encoder wrapper with projection layers |
| `configs/model_config.yaml` | Encoder choice, projection dimensions, patch size |

---

## Common Pitfalls

1. **Not freezing the encoder properly:** If you forget `param.requires_grad = False` or call `encoder.train()` instead of `encoder.eval()`, the encoder weights will update during training. This wastes GPU memory (stores gradients for ~86M parameters) and degrades performance by overfitting.

2. **Wrong image normalization:** Each pretrained encoder expects a specific normalization (mean and std per channel). SigLIP and DINOv2 use different normalization statistics. Using the wrong normalization produces garbage features. Always use the processor/transform that matches the encoder.

3. **Confusing patch count with spatial resolution:** With 16x16 patches on a 128x128 image, you get an 8x8 grid of patch features — each patch "sees" a 16x16 pixel region. This is much coarser than pixel-level resolution. For EchoTask's tasks (3cm blocks on a table), this is sufficient, but be aware of the granularity.

4. **Ignoring the CLS token vs. patch tokens distinction:** Some tutorials extract only the CLS token. For the scene perceiver, you need the patch tokens (spatial features). Make sure your extraction code returns both.

5. **Memory issues with large batches:** Even with a frozen encoder, the forward pass through a ViT-B on 128x128 images uses ~1GB per batch of 64. On a 16GB M1 Pro, you can fit ~8-10 batches simultaneously. Be mindful of memory when debugging.

---

## Further Reading

- [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) — The original Vision Transformer paper
- [SigLIP (Zhai et al., 2023)](https://arxiv.org/abs/2303.15343) — Sigmoid loss for language-image pretraining, explaining why it outperforms CLIP's contrastive loss
- [DINOv2 (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193) — Self-supervised vision features, excellent spatial understanding
- [A Survey of Visual Transformers](https://arxiv.org/abs/2111.06091) — Comprehensive overview of ViT variants and their applications
- **Previous phase:** [Phase 2: Data Pipeline](phase-02-data-pipeline.md)
- **Next phase:** [Phase 4: Architecture](phase-04-architecture.md) — Building the demonstration encoder, scene perceiver, and policy network
