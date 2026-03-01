# Phase 5: Training a Model

**Milestone:** M3 — "It Learns on Training Scenes" (second half)
**Prerequisites:** [Phase 4: Architecture](phase-04-architecture.md) complete — full model architecture implemented, forward pass verified
**Deliverables:** Training loop running, behavioral cloning converging, >80% success on training scenes for pick-and-place

---

## Overview

You have a model and you have data. Now you teach the model to imitate the expert. This phase covers **behavioral cloning** — the simplest form of imitation learning — along with the loss functions, optimizer configuration, and practical training loop engineering needed to get it working.

By the end, your model can watch a demonstration and replicate the task in scenes it has seen during training. Generalization to new scenes comes in Phase 6.

---

## Concepts

### What Is Behavioral Cloning?

Behavioral cloning is supervised learning applied to actions. The idea is dead simple:

1. You have a dataset of (observation, expert_action) pairs
2. You train a neural network to predict the expert's action given the observation
3. At test time, the network acts like the expert

**The analogy:** Imagine learning to drive by watching dashcam footage. At each frame, you see what the driver saw and what they did (turned left, braked, accelerated). You train a model to map frames → steering commands. Behavioral cloning is exactly this.

**For EchoTask specifically:**
- **Input:** A demonstration (sequence of frames + robot states) + the current scene image + current robot state
- **Target:** The expert's action at this timestep (dx, dy, dz, gripper)
- **Loss:** How different is the predicted action from the expert's action?

### Mean Squared Error (MSE) Loss

The primary loss function measures the squared difference between predicted and expert actions:

```
L_action = (1/4) * Σ (predicted_i - expert_i)²
```

where i ranges over the 4 action dimensions (dx, dy, dz, gripper).

**Why MSE?** Actions are continuous values. MSE is the natural loss for continuous regression — it's differentiable, smooth, and well-understood. It penalizes large errors quadratically (a 2x larger error gets 4x the penalty), which encourages the model to avoid big mistakes even if small ones persist.

**MSE's limitation:** It averages over all timesteps equally. But not all timesteps are equally important — the moment of grasping (where precision matters to ±1mm) gets the same weight as moving through empty space (where ±5mm doesn't matter). You could use weighted MSE, but in practice, standard MSE works well enough for EchoTask because the important timesteps (near objects) produce larger gradients naturally (the expert's actions change more abruptly there).

### Auxiliary Losses — Shaping the Task Embedding

The main action MSE loss trains the entire pipeline end-to-end. But the task embedding (output of the demonstration encoder) gets only indirect gradient signal — it affects the loss only through the policy network, which is many layers downstream.

Auxiliary losses provide **direct** gradient signal to the task embedding:

**Task classification loss (weight: 0.1):**
```
L_task = CrossEntropy(task_classifier(task_embedding), task_type_label)
```
Forces the task embedding to encode which task type was demonstrated (pick-and-place vs. stack).

**Object identification loss (weight: 0.1):**
```
L_object = BinaryCrossEntropy(object_identifier(task_embedding), color_labels)
```
Forces the task embedding to encode which object colors were involved.

**Total loss:**
```
L_total = L_action + 0.1 * L_task + 0.1 * L_object
```

The 0.1 weights are intentional — auxiliary losses are helpers, not the main objective. If you weight them too high (say 1.0), the task embedding optimizes for classification at the expense of producing useful features for the policy.

### AdamW Optimizer

**Adam** (Adaptive Moment Estimation) is the standard optimizer for deep learning. It maintains per-parameter learning rates that adapt based on gradient history:

- Parameters with consistently large gradients get smaller effective learning rates (prevents overshooting)
- Parameters with small or noisy gradients get larger effective learning rates (ensures progress)

**AdamW** adds **weight decay** — a regularization term that gently pushes all weights toward zero. This prevents any single weight from growing too large, which reduces overfitting.

Key hyperparameters:
- **Learning rate: 1e-4** — How big each parameter update is. 1e-4 is a safe starting point for Transformer models.
- **Weight decay: 1e-4** — Regularization strength. Small enough to not interfere with learning, large enough to prevent weight explosion.
- **Betas: (0.9, 0.999)** — Momentum terms. Defaults are almost always fine.

### Cosine Learning Rate Schedule

Instead of a fixed learning rate, you decay it smoothly from 1e-4 to 1e-6 over the course of training:

```
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))
```

where t is the current step and T is the total number of steps.

**Why cosine decay?**

- **Early training:** High learning rate enables rapid learning and exploration of the loss landscape
- **Mid training:** Gradually decreasing rate allows finer parameter adjustments
- **Late training:** Very low rate enables fine-tuning without overshooting good solutions

This is strictly better than a fixed learning rate and requires no manual schedule tuning.

### The Training Loop — Step by Step

A single training epoch:

```
For each batch in DataLoader:
    1. Load batch: demo_frames, demo_ee_states, current_image, robot_state, target_action, task_type

    2. Forward pass:
       a. task_embedding = demo_encoder(vision_encoder(demo_frames), demo_ee_states)
       b. scene_features = scene_perceiver(vision_encoder(current_image))
       c. predicted_action = policy(task_embedding, scene_features, robot_state)
       d. task_logits, object_logits = aux_heads(task_embedding)

    3. Compute losses:
       a. L_action = MSE(predicted_action, target_action)
       b. L_task = CrossEntropy(task_logits, task_type_label)
       c. L_object = BCE(object_logits, color_labels)
       d. L_total = L_action + 0.1 * L_task + 0.1 * L_object

    4. Backward pass:
       optimizer.zero_grad()
       L_total.backward()
       torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
       optimizer.step()
       scheduler.step()

    5. Log metrics (loss, learning rate, gradient norms)
```

**Gradient clipping** (`clip_grad_norm_` with max_norm=1.0) prevents training instability from occasional exploding gradients. This is especially important early in training when the model's predictions are random and losses are large.

### Cloud GPU Training

The model won't train efficiently on your M1 Pro — it'll take days instead of hours. You need a cloud GPU.

**Options:**
- **Google Colab Pro ($10/month):** A100 access (intermittent), easiest setup
- **Lambda Cloud (~$1.50/hr for A100):** Dedicated instances, more reliable
- **RunPod, Vast.ai:** Cheaper spot instances, less reliable

**Workflow:**
1. Develop and debug locally on the M1 Pro with a tiny dataset (10 demos, 5 epochs)
2. Upload the full dataset and code to the cloud instance
3. Run the full training (100-200 epochs, 8-12 hours on A100)
4. Download the trained model checkpoint
5. Evaluate locally

**A100 memory budget (40GB):**

| Component | Memory |
|-----------|--------|
| Frozen vision encoder (ViT-B) | ~350 MB |
| Trainable model parameters | ~100 MB |
| Optimizer states (AdamW: 2x parameters) | ~200 MB |
| Gradients | ~100 MB |
| Batch data (batch_size=64, images + features) | ~2 GB |
| Activations for backward pass | ~3-5 GB |
| **Total** | **~6-8 GB** |

Plenty of headroom. You could increase batch size to 128 or 256 if training is slow.

---

## Implementation Guide

### Step 1: Create the Training Configuration

Define hyperparameters in `configs/train_config.yaml`:

```yaml
# Training
batch_size: 64
num_epochs: 150
learning_rate: 1.0e-4
min_learning_rate: 1.0e-6
weight_decay: 1.0e-4
grad_clip_norm: 1.0

# Loss weights
action_loss_weight: 1.0
task_class_loss_weight: 0.1
object_id_loss_weight: 0.1

# Data
num_workers: 4
demo_subsample_rate: 5  # Use every 5th frame from demo
max_demo_frames: 60

# Checkpointing
save_every_epochs: 10
eval_every_epochs: 5
```

### Step 2: Implement the Training Loop

Create `backend/training/bc_trainer.py`:

1. **Initialization:** Load config, create model, optimizer (AdamW), scheduler (CosineAnnealingLR), dataset, dataloader
2. **Train loop:** Iterate over epochs and batches, compute forward pass, losses, backward pass, optimizer step
3. **Validation:** Every N epochs, run the model on a held-out validation set (10% of demos) and log validation loss
4. **Checkpointing:** Save model weights every N epochs and keep the best checkpoint (lowest validation loss)
5. **Logging:** Track training loss, validation loss, learning rate, gradient norms, per-component losses

```python
class BCTrainer:
    def __init__(self, config, model, dataset):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.trainable_parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=config.min_learning_rate
        )
        self.dataloader = DataLoader(
            dataset, batch_size=config.batch_size,
            shuffle=True, num_workers=config.num_workers,
            pin_memory=True
        )

    def train_epoch(self):
        self.model.train()
        epoch_losses = []
        for batch in self.dataloader:
            loss = self.train_step(batch)
            epoch_losses.append(loss)
        return np.mean(epoch_losses)

    def train_step(self, batch):
        # Forward
        task_emb = self.model.encode_demonstration(
            batch["demo_frames"], batch["demo_ee_states"]
        )
        action = self.model(task_emb, batch["current_image"], batch["robot_state"])
        task_logits, obj_logits = self.model.aux_heads(task_emb)

        # Losses
        action_loss = F.mse_loss(action, batch["target_action"])
        task_loss = F.cross_entropy(task_logits, batch["task_type"])
        object_loss = F.binary_cross_entropy_with_logits(obj_logits, batch["color_labels"])
        total_loss = action_loss + 0.1 * task_loss + 0.1 * object_loss

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return total_loss.item()
```

### Step 3: Implement Validation and Checkpointing

Add to the trainer:

- **Validation loss:** Run forward pass (no backward pass, `torch.no_grad()`) on held-out demos
- **Model checkpointing:** Save `model.state_dict()` to disk every N epochs
- **Best model tracking:** Keep the checkpoint with the lowest validation action loss
- **Early stopping (optional):** Stop if validation loss hasn't improved in 30 epochs

### Step 4: Add In-Simulation Evaluation

Validation loss tells you how well the model predicts actions, but not whether those actions actually complete tasks. Add periodic in-simulation evaluation:

1. Every 20 epochs, run the model in the simulation on 20 training scenes
2. Measure task success rate (did the robot complete the task?)
3. Log the success rate alongside the loss curves

This connects the abstract loss number to actual robot performance.

### Step 5: Create the Training Script

Create `scripts/train.py` that:

1. Parses command-line arguments (config path, data path, output path, GPU device)
2. Loads the dataset
3. Creates the model
4. Initializes the trainer
5. Runs the training loop
6. Saves the final model

```bash
# Local debugging (tiny dataset)
python scripts/train.py --config configs/train_config.yaml --data data/demos/ --epochs 5 --batch_size 4

# Full training (cloud GPU)
python scripts/train.py --config configs/train_config.yaml --data data/demos/ --output checkpoints/
```

### Step 6: Monitor and Debug Training

Key things to watch during training:

**Healthy training signs:**
- Training loss decreases steadily over the first 20 epochs
- Validation loss tracks training loss (no large gap)
- Gradient norms are stable (not growing or oscillating wildly)
- Learning rate follows the cosine curve

**Warning signs and fixes:**

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Loss doesn't decrease at all | Learning rate too low, or data loading bug | Increase LR to 1e-3, verify data labels match images |
| Loss decreases then explodes | Learning rate too high, no gradient clipping | Reduce LR, add/verify gradient clipping |
| Training loss drops, validation loss rises | Overfitting | Add more data augmentation, increase weight decay |
| Action predictions are all near zero | Dead ReLU or tanh saturation | Check activation distributions, try LeakyReLU |
| Auxiliary losses don't decrease | Task embedding isn't learning semantics | Increase auxiliary loss weights temporarily to 0.5 |

---

## Learning Checkpoints

**Conceptual checks:**

- [ ] Why is behavioral cloning just supervised learning? What's the "label" for each example?
- [ ] Why use MSE for actions instead of classification (binning actions into discrete categories)?
- [ ] What do the auxiliary losses contribute that the main action loss doesn't?
- [ ] Why does cosine learning rate decay outperform a fixed learning rate?
- [ ] Why is gradient clipping important, especially early in training?
- [ ] What's the difference between training loss and validation loss, and what does a gap between them indicate?

**Practical milestones:**

- [ ] Training loop runs without errors on a small local dataset (10 demos, 5 epochs)
- [ ] Loss decreases over epochs — model is learning
- [ ] Training completes on cloud GPU (100-200 epochs, 8-12 hours on A100)
- [ ] Validation loss tracks training loss (no severe overfitting)
- [ ] In-simulation evaluation: >80% success rate on training scenes for pick-and-place
- [ ] Model checkpoint saved and loadable
- [ ] Video recording of the trained model succeeding on 10 consecutive training scenes

---

## Key Files

| File | Purpose |
|------|---------|
| `backend/training/bc_trainer.py` | Behavioral cloning training loop |
| `configs/train_config.yaml` | Training hyperparameters |
| `scripts/train.py` | CLI script to launch training |
| `checkpoints/` | Saved model weights |

---

## Common Pitfalls

1. **Not separating frozen and trainable parameters:** When creating the optimizer, only pass trainable parameters. If you pass `model.parameters()` including the frozen vision encoder, the optimizer allocates unnecessary state (momentum, variance) for 86M frozen parameters, wasting ~700MB of memory.

2. **DataLoader num_workers on macOS:** On macOS with MPS (Metal Performance Shaders), `num_workers > 0` can cause issues with HDF5 file handles. If you encounter `HDF5 error: unable to open file`, try `num_workers=0` or switch to opening/closing HDF5 files per access.

3. **Forgetting model.eval() during validation:** If dropout or batch normalization layers are present, calling `model.train()` during validation applies dropout, making validation metrics noisy and unreliable.

4. **Action scale mismatch between training and evaluation:** If training targets are in [-0.05, 0.05] but the simulation expects [-1, 1] (or vice versa), the robot will either barely move or wildly overshoot. Verify the action scale is consistent.

5. **Not using pin_memory with GPU training:** Setting `pin_memory=True` in the DataLoader speeds up CPU-to-GPU data transfer by 2-3x. Free performance.

6. **Checkpoint only saves model weights, not optimizer state:** If you need to resume training after a crash, you need both `model.state_dict()` and `optimizer.state_dict()`. Save both.

---

## Further Reading

- [A Reduction of Imitation Learning to No-Regret Online Learning (Ross et al., 2011)](https://arxiv.org/abs/1011.0686) — Theoretical analysis of behavioral cloning and its failure modes (motivates DAgger in Phase 6)
- [An Introduction to Deep Reinforcement Learning (Francois-Lavet et al., 2018)](https://arxiv.org/abs/1811.12560) — Chapter 3 covers imitation learning in detail
- [Decoupled Weight Decay Regularization (AdamW paper, Loshchilov & Hutter, 2017)](https://arxiv.org/abs/1711.05101) — Why AdamW is better than Adam with L2 regularization
- [PyTorch Training Loop Best Practices](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html) — Official tutorial on training loops, validation, and checkpointing
- **Previous phase:** [Phase 4: Architecture](phase-04-architecture.md)
- **Next phase:** [Phase 6: Generalization](phase-06-generalization.md) — DAgger refinement and evaluation on held-out scenes
