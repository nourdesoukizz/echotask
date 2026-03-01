# Phase 4: Architecture — The Agent

**Milestone:** M3 — "It Learns on Training Scenes" (first half)
**Prerequisites:** [Phase 3: Computer Vision](phase-03-computer-vision.md) complete — vision encoder integrated, feature extraction working
**Deliverables:** Full architecture implemented (demonstration encoder + scene perceiver + policy network), forward pass verified on dummy data

---

## Overview

This is where EchoTask becomes an **agent** — a system that perceives, reasons, and acts. You'll build the three modules that form the core pipeline:

1. **Demonstration Encoder** — watches a demo and produces a task embedding ("what should I do?")
2. **Scene Perceiver** — looks at the current scene and produces spatial features ("what do I see?")
3. **Policy Network** — combines task intent with scene perception to output actions ("how do I move?")

By the end of this phase, the full model accepts a demonstration + current observation and outputs an action vector. No training yet — just architecture verification.

---

## Concepts

### What Makes This an "Agent"?

An agent is a system that:

1. **Perceives** its environment (camera image → scene features)
2. **Has a goal** (demonstration → task embedding)
3. **Acts** to achieve the goal (policy → motor commands)
4. **Operates in a loop** (perceive → act → perceive → act → ...)

EchoTask's agent is a **reactive policy** — it doesn't plan ahead or search over future states. At each timestep, it looks at the current scene, consults the task embedding, and outputs the next small action. The "intelligence" is in the learned mapping from (task + scene) → action, not in explicit planning.

This is analogous to how a skilled human picks up objects — you don't consciously plan every millimeter of hand movement. Your visual cortex (scene perceiver) and motor cortex (policy) coordinate through learned reflexes. The demonstration serves as the "instruction" your prefrontal cortex provides.

### The Demonstration Encoder — Temporal Understanding

The demonstration encoder must compress a variable-length video (30-60 frames) into a fixed-length vector that captures task intent. This is a **sequence-to-vector** problem.

**Why a Transformer for temporal aggregation?**

The demonstration is a sequence of frames showing a task unfolding over time. The order matters — "pick up red, then place on blue" is different from "pick up blue, then place on red." A Transformer's self-attention mechanism can model these temporal dependencies:

- Frame 10 (approaching the red block) attends to Frame 25 (placing on the blue block) to understand the full task arc
- The model learns which frames are important (the grasp moment, the placement moment) and which are filler (moving through empty space)

**Architecture details:**

```
Input: N frames, each with (visual_feature_768d + ee_state_4d)

1. Visual features (768d) → Linear projection → 256d
2. Robot state (4d) → Linear projection → 256d
3. Concatenate → 512d per frame (or add them for 256d)
4. Add learned positional embeddings (so the model knows frame order)
5. Pass through Transformer encoder (4-6 layers, 256d, 4-8 heads)
6. Pool the output sequence into a single vector:
   - Option A: Take the mean of all output tokens
   - Option B: Add a learnable [TASK] token, use its final output
7. Output: task_embedding (256d or 512d)
```

**What the task embedding should encode:**

Think of the task embedding as a compressed instruction: "pick up the red thing and put it on top of the blue thing." It should NOT encode the specific positions from the demo (those are irrelevant in the new scene). This is the hardest representation learning challenge in the project — the auxiliary losses in Phase 5 help shape this.

### The Scene Perceiver — Spatial Understanding

The scene perceiver is simpler than the demonstration encoder. It takes the current camera image and produces a set of **spatially-organized features** that the policy can attend to.

**Architecture:**

```
Input: Current scene image (128x128 RGB)

1. Frozen vision encoder (SigLIP/DINOv2) → patch features (64 patches, 768d each)
2. Linear projection → (64 patches, 256d each)
3. Output: scene_features (64, 256d)
```

That's it — the scene perceiver is primarily the frozen vision encoder plus a learned projection. The heavy lifting is done by the pretrained encoder. The projection layer adapts the generic features into EchoTask's embedding space.

**Why keep spatial structure?** The 64 patch features maintain an 8x8 spatial grid. The policy network can cross-attend to specific spatial locations: "attend to the patch where the red block is" → "that's at position (3, 2) in the grid" → "move the arm toward that direction."

### The Policy Network — Cross-Attention and Action Generation

The policy network is where task intent meets scene perception. It takes three inputs:

1. **Task embedding** (256d) — what to do
2. **Scene features** (64 x 256d) — what the scene looks like spatially
3. **Current robot state** (4d: xyz + gripper) — where the arm currently is

**Architecture (cross-attention transformer decoder):**

```
Inputs:
  - task_embedding: (256d)
  - scene_features: (64, 256d)
  - robot_state: (4d) → Linear → (256d)

1. Create the "query" by combining task embedding + robot state:
   query = Linear(concat(task_embedding, robot_state_projected))  → (256d)
   Expand to 1 or a few query tokens

2. Cross-attention transformer decoder (4 layers):
   - Query: task-conditioned robot state tokens
   - Key/Value: scene features (64 spatial tokens)
   - Each layer: the query attends to the scene features,
     focusing on task-relevant spatial locations

3. Final MLP head:
   - Input: decoder output (256d)
   - Hidden: 256 → 128 (ReLU)
   - Output: 4d action vector (dx, dy, dz, gripper)
   - Activation: tanh (bounds actions to [-1, 1], then scale to actual range)

4. Optional: 5th output for termination signal (sigmoid, threshold at 0.5)
```

**Why cross-attention?**

Cross-attention lets the policy **selectively focus** on relevant parts of the scene based on the task:

- If the task embedding encodes "pick up the red block," the cross-attention learns to attend strongly to the patch feature that corresponds to the red block's location
- If the robot has already grasped the red block and needs to place it on the blue block, the attention shifts to the blue block's patch

This is fundamentally different from simply concatenating all inputs — concatenation treats all scene locations equally, while cross-attention learns a task-dependent spatial focus.

### The Full Forward Pass

Here's the complete data flow through EchoTask at inference time:

```
DEMONSTRATION (recorded earlier):
  demo_frames (N, 128, 128, 3) ──→ vision_encoder ──→ (N, 768)
  demo_ee_states (N, 4)         ──→ linear          ──→ (N, 256)
                                     ↓ concat + positional encoding
                                temporal_transformer (4-6 layers)
                                     ↓ pool
                                task_embedding (256d)

CURRENT SCENE (live, every timestep):
  scene_image (128, 128, 3) ──→ vision_encoder ──→ (64, 768)
                                     ↓ linear projection
                                scene_features (64, 256d)

POLICY (every timestep):
  task_embedding (256d)    ──┐
  scene_features (64, 256) ──┤──→ cross_attention_decoder ──→ MLP ──→ action (4d)
  robot_state (4d)         ──┘
```

**Timing:** The demonstration encoder runs **once** per episode (when the demo is provided). The scene perceiver + policy run **every timestep** (10 Hz). This is efficient — the expensive demo encoding happens once, and the per-step computation is lightweight.

### Auxiliary Heads (Training Only)

Two small MLP heads attach to the task embedding during training:

**Task classification head:**
```
task_embedding (256d) → Linear(256, 128) → ReLU → Linear(128, num_task_types) → softmax
```
Predicts the task type (pick-and-place, stack, sort, etc.). Cross-entropy loss, weighted at 0.1x the main loss.

**Object identification head:**
```
task_embedding (256d) → Linear(256, 128) → ReLU → Linear(128, num_colors) → sigmoid (per color)
```
Predicts which object colors were involved in the task. Multi-label binary cross-entropy, weighted at 0.1x the main loss.

These heads are discarded after training. Their only purpose is to provide gradient signal that shapes the task embedding to encode semantic information (task type, relevant objects).

---

## Implementation Guide

### Step 1: Implement the Demonstration Encoder

Create `backend/models/demo_encoder.py`:

```python
class DemonstrationEncoder(nn.Module):
    def __init__(self, visual_dim=768, state_dim=4, embed_dim=256,
                 num_layers=4, num_heads=4, max_demo_len=64):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, embed_dim)
        self.state_proj = nn.Linear(state_dim, embed_dim)
        self.combine = nn.Linear(embed_dim * 2, embed_dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, max_demo_len, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, visual_features, ee_states, mask=None):
        # visual_features: (B, N, 768) — from frozen encoder
        # ee_states: (B, N, 4) — end-effector xyz + gripper
        v = self.visual_proj(visual_features)
        s = self.state_proj(ee_states)
        x = self.combine(torch.cat([v, s], dim=-1))

        # Add positional embeddings
        N = x.shape[1]
        x = x + self.pos_embedding[:, :N, :]

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)

        # Pool to single vector (mean pooling)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)
            lengths = (~mask).sum(dim=1, keepdim=True)
            task_embedding = x.sum(dim=1) / lengths
        else:
            task_embedding = x.mean(dim=1)

        return self.output_norm(task_embedding)  # (B, 256)
```

### Step 2: Implement the Scene Perceiver

Create `backend/models/scene_perceiver.py`:

```python
class ScenePerceiver(nn.Module):
    def __init__(self, visual_dim=768, embed_dim=256):
        super().__init__()
        self.spatial_proj = nn.Linear(visual_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, patch_features):
        # patch_features: (B, num_patches, 768) — from frozen encoder
        scene_features = self.spatial_proj(patch_features)
        return self.norm(scene_features)  # (B, 64, 256)
```

### Step 3: Implement the Policy Network

Create `backend/models/policy.py`:

```python
class PolicyNetwork(nn.Module):
    def __init__(self, embed_dim=256, state_dim=4, num_layers=4,
                 num_heads=4, action_dim=4):
        super().__init__()
        self.state_proj = nn.Linear(state_dim, embed_dim)
        self.query_proj = nn.Linear(embed_dim * 2, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # Bound to [-1, 1]
        )

        self.action_scale = 0.05  # Scale to [-0.05, 0.05] meters

    def forward(self, task_embedding, scene_features, robot_state):
        # task_embedding: (B, 256)
        # scene_features: (B, 64, 256)
        # robot_state: (B, 4)

        state_feat = self.state_proj(robot_state)
        query = self.query_proj(torch.cat([task_embedding, state_feat], dim=-1))
        query = query.unsqueeze(1)  # (B, 1, 256)

        decoded = self.decoder(query, scene_features)  # (B, 1, 256)
        decoded = decoded.squeeze(1)  # (B, 256)

        action = self.action_head(decoded)  # (B, 4) in [-1, 1]

        # Scale position deltas, leave gripper in [-1, 1]
        action_scaled = action.clone()
        action_scaled[:, :3] = action[:, :3] * self.action_scale

        return action_scaled
```

### Step 4: Implement Auxiliary Heads

Add auxiliary classification heads (used during training only):

```python
class AuxiliaryHeads(nn.Module):
    def __init__(self, embed_dim=256, num_task_types=4, num_colors=8):
        super().__init__()
        self.task_classifier = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(), nn.Linear(128, num_task_types)
        )
        self.object_identifier = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(), nn.Linear(128, num_colors)
        )

    def forward(self, task_embedding):
        task_logits = self.task_classifier(task_embedding)
        object_logits = self.object_identifier(task_embedding)
        return task_logits, object_logits
```

### Step 5: Assemble the Full Model

Create a top-level `EchoTaskModel` that combines all components:

```python
class EchoTaskModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_encoder = VisionFeatureExtractor(...)  # From Phase 3
        self.demo_encoder = DemonstrationEncoder(...)
        self.scene_perceiver = ScenePerceiver(...)
        self.policy = PolicyNetwork(...)
        self.aux_heads = AuxiliaryHeads(...)  # Only used during training

    def encode_demonstration(self, demo_frames, demo_ee_states):
        """Run once per episode."""
        with torch.no_grad():
            visual_features = self.vision_encoder.extract_global(demo_frames)
        task_embedding = self.demo_encoder(visual_features, demo_ee_states)
        return task_embedding

    def forward(self, task_embedding, current_image, robot_state):
        """Run every timestep."""
        with torch.no_grad():
            _, patch_features = self.vision_encoder(current_image)
        scene_features = self.scene_perceiver(patch_features)
        action = self.policy(task_embedding, scene_features, robot_state)
        return action
```

### Step 6: Verify with Dummy Data

Write a test that creates the full model and runs a forward pass:

```python
# Dummy inputs
demo_frames = torch.randn(2, 30, 3, 128, 128)    # Batch=2, 30 demo frames
demo_ee_states = torch.randn(2, 30, 4)             # xyz + gripper per frame
current_image = torch.randn(2, 3, 128, 128)        # Current scene
robot_state = torch.randn(2, 4)                     # Current ee state

# Forward pass
task_emb = model.encode_demonstration(demo_frames, demo_ee_states)
action = model(task_emb, current_image, robot_state)

assert task_emb.shape == (2, 256)
assert action.shape == (2, 4)
assert action[:, :3].abs().max() <= 0.05  # Position deltas bounded
```

Count trainable parameters — should be ~15-25M (excluding the frozen vision encoder).

---

## Learning Checkpoints

**Conceptual checks:**

- [ ] Why does the demonstration encoder use a Transformer instead of an RNN or simple averaging?
- [ ] What does cross-attention achieve that concatenation doesn't?
- [ ] Why is the task embedding a fixed-size vector, regardless of demo length?
- [ ] Why do auxiliary heads help the task embedding quality, even though they're discarded after training?
- [ ] Why does the demonstration encoder run once per episode while the policy runs every timestep?

**Practical milestones:**

- [ ] Demonstration encoder: accepts (B, N, 768+4) → outputs (B, 256) task embedding
- [ ] Scene perceiver: accepts (B, 64, 768) → outputs (B, 64, 256) scene features
- [ ] Policy: accepts task embedding + scene features + robot state → outputs (B, 4) action
- [ ] Full forward pass succeeds on dummy data with correct shapes
- [ ] Auxiliary heads produce valid logits from the task embedding
- [ ] Trainable parameter count is 15-25M
- [ ] Model fits in GPU memory (A100 40GB) with batch size 64

---

## Key Files

| File | Purpose |
|------|---------|
| `backend/models/demo_encoder.py` | Demonstration → task embedding (temporal transformer) |
| `backend/models/scene_perceiver.py` | Current image → spatial scene features |
| `backend/models/policy.py` | Task embedding + scene → action output |
| `backend/models/vision_encoder.py` | Frozen pretrained encoder (from Phase 3) |
| `configs/model_config.yaml` | Architecture hyperparameters (dims, layers, heads) |

---

## Common Pitfalls

1. **Forgetting padding masks in the demo encoder:** Demonstrations have variable lengths. If you batch demos of length 20 and 50 together, the shorter one is padded. Without a padding mask, the Transformer attends to padding tokens, corrupting the task embedding.

2. **Wrong attention direction in cross-attention:** In `nn.TransformerDecoder`, the first argument is the query (task/robot tokens) and the second is the key/value (scene features). Swapping them produces a model that compiles but learns poorly.

3. **Tanh saturation:** If the policy outputs are consistently near ±1 (tanh saturation), gradients vanish. Monitor the pre-tanh activations during training. If they're consistently >3 or <-3, the model is saturating.

4. **Positional embedding shape mismatch:** If the demo length exceeds `max_demo_len`, the positional embedding lookup fails. Set `max_demo_len` generously (e.g., 128) and subsample long demos to fit.

5. **Not sharing the vision encoder:** The demonstration frames and the current scene image should use the same frozen encoder instance. If you accidentally create two separate encoder copies, you double memory usage and lose feature compatibility.

6. **Action scaling applied twice:** If you scale actions in both the policy network (tanh * 0.05) and in the environment step function, the robot will barely move. Pick one place to apply scaling.

---

## Further Reading

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — The original Transformer paper; focus on the encoder and cross-attention mechanisms
- [BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning (Jang et al., 2021)](https://arxiv.org/abs/2202.02005) — One-shot imitation learning with task conditioning, architecturally similar to EchoTask
- [One-Shot Imitation Learning (Duan et al., 2017)](https://arxiv.org/abs/1703.07326) — The foundational one-shot imitation learning paper from OpenAI
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Excellent visual guide to Transformer internals
- **Previous phase:** [Phase 3: Computer Vision](phase-03-computer-vision.md)
- **Next phase:** [Phase 5: Training](phase-05-training.md) — Training the model with behavioral cloning
