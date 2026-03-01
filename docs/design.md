# EchoTask — Design Document

## One-Shot Imitation Learning for Robotic Manipulation

**Author:** Nour
**Status:** Draft
**Last Updated:** March 2026

---

## 1. What Is EchoTask

EchoTask is a system where a simulated robot arm watches a single human demonstration of a manipulation task, then replicates that task autonomously in a completely different scene configuration — different object positions, colors, and layouts.

The system takes two inputs: a recorded demonstration (a sequence of image frames + robot actions from a human performing a task) and a live camera view of a new scene. It outputs a sequence of robot motor commands that accomplish the same task in the new scene.

**The demo moment:** A split-screen video. Left: the human demonstrating "pick up the red block and place it on the blue one." Right: the robot autonomously doing the same thing in a brand new scene arrangement it has never seen, in real time.

---

## 2. Scope and Constraints

### What EchoTask Does

- Accepts a single demonstration of a tabletop manipulation task
- Understands the intent of the demonstration (what was done and to which objects)
- Perceives a new scene through a virtual camera
- Executes the same task in the new scene, adapting to different object positions and arrangements

### What EchoTask Does Not Do

- Work on a physical robot (simulation only)
- Handle tasks it has never seen a category of during training (it generalizes across scene layouts, not across entirely novel task types)
- Work from natural language instructions (the input is a visual demonstration, not text)
- Operate in cluttered or highly complex environments (tabletop with 3-6 simple objects)

### Hardware Constraints

- All development done on Apple M1 Pro (16/32GB unified memory)
- Training runs executed on cloud GPU (Google Colab Pro or Lambda — A100)
- No physical robot or sensors required

---

## 3. Task Definitions

EchoTask will support the following manipulation tasks, in order of difficulty:

### Tier 1 — Starter Tasks (build these first)

- **Pick and Place:** Pick up object A, place it at location B
- **Stack:** Pick up object A, stack it on top of object B

### Tier 2 — Intermediate Tasks (add after Tier 1 works)

- **Sort:** Move objects to designated zones by color or shape
- **Reorder:** Rearrange a sequence of objects into a demonstrated order

### Tier 3 — Stretch Tasks (only if Tier 2 generalizes well)

- **Pour:** Tilt a container to pour contents into a target vessel
- **Insert:** Place an object into a slot or receptacle

Each task must work across randomized scene configurations: object positions are randomized within the reachable workspace, object colors are varied, and the camera viewpoint has slight perturbations.

---

## 4. System Architecture — The Three Modules

EchoTask has three major modules that work together sequentially.

### 4.1 The Demonstration Encoder (Watching)

**Purpose:** Take a full demonstration video and compress it into a fixed-length "task embedding" — a vector that encodes what the human did, to which objects, and in what order.

**Input:** A sequence of image frames from the demonstration (captured at 5-10 fps during recording) plus the corresponding robot end-effector states (position, gripper open/close) at each frame.

**Processing steps:**

1. Each image frame passes through a frozen vision encoder (SigLIP or DINOv2) to produce a visual feature vector per frame
2. The robot state at each frame (end-effector xyz position + gripper state) is concatenated with the visual feature
3. The full sequence of (visual feature + robot state) vectors passes through a temporal aggregation network — a lightweight transformer encoder (4-6 layers, 256-dim) with learned positional embeddings
4. The output is a single fixed-length task embedding vector (256 or 512 dimensions) that summarizes the entire demonstration

**Key design decision:** The demonstration encoder must learn to ignore irrelevant details (exact pixel positions, lighting) and capture task-level intent (which object was manipulated, where it went, in what order). This is the hardest representation learning challenge in the project.

**What the task embedding should capture:**

- Which object(s) were interacted with (identity by appearance)
- What action was performed (pick, place, stack)
- The spatial relationship of the goal state (on top of, next to, inside)

**What the task embedding should NOT capture:**

- Exact coordinates from the demonstration scene (these are meaningless in the new scene)
- The specific trajectory taken (only the goal matters, not the path)
- Visual details unrelated to the task (background, shadows)

### 4.2 The Scene Perceiver (Seeing)

**Purpose:** Take the current camera image of the new scene and produce a structured representation of what objects are present and where they are.

**Input:** A single RGB image from the virtual camera looking at the new scene.

**Processing steps:**

1. The same frozen vision encoder (SigLIP or DINOv2) processes the current scene image into a grid of visual feature tokens (patch-level features, not just a single vector)
2. These patch features preserve spatial information — the model knows not just what objects exist but where in the image they are
3. The patch features are projected into the same embedding space as the task embedding through a learned linear projection

**Key design decision:** Using the same vision encoder backbone for both the demonstration and the live scene ensures the feature spaces are compatible. The model can match "the red block I saw in the demo" to "the red block I see now" because they're encoded by the same network.

### 4.3 The Policy Network (Acting)

**Purpose:** Take the task embedding (from the demonstration) and the scene features (from the current view), and output the next robot action.

**Input:** Task embedding vector + current scene patch features + current robot end-effector state (xyz + gripper)

**Processing steps:**

1. The task embedding is broadcast and concatenated with each scene patch feature, creating task-conditioned scene features. Each spatial location in the scene is now "aware" of what task needs to be performed
2. These task-conditioned features, along with the current robot state, pass through a small transformer decoder (4 layers) that cross-attends to the scene features
3. The final output is a 4-dimensional action vector: (delta-x, delta-y, delta-z, gripper-command) representing the incremental movement of the end-effector

**Action space details:**

- delta-x, delta-y, delta-z: continuous values in range [-0.05, 0.05] meters, representing small incremental movements per timestep
- gripper-command: continuous value in [-1, 1] where -1 is fully open and +1 is fully closed
- The policy runs at 10 Hz — it outputs 10 action commands per second, each producing a small movement

**Key design decision:** Using incremental (delta) actions rather than absolute target positions makes the policy more robust to different scene configurations. The robot takes small steps, re-perceives the scene at each step, and adjusts. This closed-loop execution is critical for generalization.

**Closed-loop execution flow:**

1. Camera captures current scene image
2. Scene perceiver encodes it
3. Policy takes (task embedding + scene features + current robot state) → outputs action
4. Action is applied to the robot in simulation
5. Simulation steps forward
6. Go to step 1
7. Repeat until the policy outputs a "terminate" signal (a 5th output dimension, thresholded) or a maximum number of steps is reached

---

## 5. Where the LLM Fits In

The LLM is not used during real-time execution. It plays two roles in the system:

### 5.1 Task Intent Extraction (Training Phase)

During the data generation phase, a fine-tuned LLM (Llama 3 8B with LoRA) is used to automatically label demonstrations with structured task descriptions. Given a sequence of image frames from a demonstration, the LLM produces a structured annotation:

- Task type: "pick_and_place"
- Source object: "red block"
- Target: "on top of blue block"
- Key waypoints: ["approach red block", "grasp", "lift", "move to blue block", "release"]

These structured labels are used as auxiliary training signal — the demonstration encoder is trained not only to produce embeddings that lead to successful imitation, but also to produce embeddings from which the LLM-generated task description can be reconstructed (via a small auxiliary decoder head). This encourages the task embedding to capture semantic task intent.

### 5.2 Demo Narration (Inference / Demo Phase)

During the live demo, the LLM can narrate what the system is doing in real time — "I observed the human pick up the red block and place it on the blue one. I can see the red block is now on the far left. Planning approach trajectory..." This is for the "holy shit" factor of the demo, not for the core functionality.

### Fine-tuning Details

- Base model: Llama 3 8B (or Phi-3 Mini 3.8B if memory is tight)
- Method: LoRA (rank 16, alpha 32) on attention layers
- Training data: ~5,000 (image sequence, structured task label) pairs, generated semi-automatically using a larger model (Claude or GPT-4V) to bootstrap annotations, then cleaned manually
- Training compute: ~4-6 hours on a single A100

---

## 6. Simulation Environment

### 6.1 Simulator Choice: MuJoCo + Robosuite

**MuJoCo** is the physics engine. It handles collision detection, gravity, friction, contact dynamics — all the physical realism.

**Robosuite** is a framework built on top of MuJoCo that provides pre-built robot models, task environments, and rendering. It saves weeks of setup compared to building from scratch.

### 6.2 Robot

**Franka Emika Panda** — a 7-degree-of-freedom robot arm with a parallel-jaw gripper. This is the standard robot in manipulation research. Robosuite has it built in with tuned control parameters.

Control mode: **Operational Space Control (OSC)** — you send Cartesian position deltas (dx, dy, dz) and the controller handles the joint-level math. This matches the policy's action space directly.

### 6.3 Table and Objects

- A flat table surface centered in the robot's workspace
- 2-6 primitive objects per scene: cubes (3cm), cylinders (3cm diameter, 5cm height), small boxes
- Objects are colored with randomized RGB values from a fixed palette of 8 distinguishable colors
- Object positions are randomized within a defined rectangular zone on the table (roughly 30cm x 40cm)
- Objects always start on the table surface (no mid-air spawning)

### 6.4 Camera Setup

- One fixed camera mounted above and behind the robot, angled downward at roughly 45 degrees
- Image resolution: 128x128 pixels (during training) and 256x256 (during demo recording)
- 128x128 is sufficient for the vision encoder and dramatically reduces memory and compute during training
- A second camera angle (front-facing or side view) is used only for the demo video, not as model input

### 6.5 Scene Randomization

Every episode (every demonstration or evaluation rollout), the following are randomized:

- Object positions on the table (uniform random within workspace bounds)
- Object colors (sampled from the 8-color palette)
- Number of distractor objects (objects not involved in the task): 0-3
- Lighting intensity: ±20% variation
- Camera position: ±2cm jitter in xyz, ±3 degrees rotation jitter

This randomization is what forces the model to generalize rather than memorize.

---

## 7. Data Collection Pipeline

### 7.1 Teleoperation Interface

You need a way to demonstrate tasks by controlling the robot with your mouse/keyboard.

**Implementation:** Robosuite has a built-in teleoperation interface using keyboard controls or a SpaceMouse. Keyboard is free and sufficient — arrow keys for xy movement, page up/down for z, spacebar for gripper toggle.

**During teleoperation, the system records:**

- RGB image from the camera at every timestep (10 Hz)
- Robot end-effector position (x, y, z) at every timestep
- Gripper state (open/close) at every timestep
- Joint positions (7 values) at every timestep

All saved as an HDF5 file per demonstration.

### 7.2 How Many Demonstrations You Need

- **Per task type:** 200-500 demonstrations across randomized scenes
- **Total for Tier 1 (pick-and-place + stack):** ~600-1000 demonstrations
- **Time per demonstration:** ~15-30 seconds of active control = roughly 150-300 timesteps
- **Time to collect:** At ~1 minute per demo (including reset time), 1000 demos = ~16 hours of collection spread across several days

This is tedious. To speed it up:

- Write a scripted policy (a simple hardcoded controller that solves the task using known object positions) to generate 80% of the demonstrations automatically
- Manually teleoperate 20% to add human-like variation and imperfection
- The scripted policy uses ground truth object positions from the simulator (cheating that's only allowed during data generation, not during evaluation)

### 7.3 Data Augmentation

Applied to demonstration images during training:

- Random color jitter (brightness, contrast, saturation: ±15%)
- Random crop and resize (crop 90-100% of image, resize back to 128x128)
- Gaussian noise injection (sigma = 0.01)
- Random horizontal flip with corresponding action mirroring (flip delta-x sign)

---

## 8. Training Strategy

### 8.1 Two-Phase Training

**Phase 1 — Behavioral Cloning (Weeks 5-6)**

The simplest approach: train the entire pipeline (demonstration encoder + scene perceiver + policy network) end-to-end to predict the expert's action at each timestep, given the demonstration and the current scene.

Loss function: Mean Squared Error between predicted action (dx, dy, dz, gripper) and the actual expert action from the recorded demonstration.

This will get you a policy that works ~60-70% of the time on training scenes and ~30-40% on new scenes. Good enough to validate the architecture before moving to Phase 2.

**Phase 2 — DAgger-Style Refinement (Weeks 7-9)**

Behavioral cloning fails when the robot drifts into states the expert never visited. The robot makes a small mistake, ends up in an unfamiliar position, and the errors compound.

DAgger (Dataset Aggregation) fixes this:

1. Run the current policy in simulation to collect rollouts
2. At each state the policy visits, query the scripted expert for what the correct action should have been
3. Add these (state, correct action) pairs to the training dataset
4. Retrain the policy on the expanded dataset
5. Repeat 3-5 times

This teaches the policy to recover from its own mistakes, dramatically improving robustness.

### 8.2 Auxiliary Losses

In addition to the main action prediction loss, two auxiliary losses improve the task embedding quality:

**Task classification loss:** A small MLP head on top of the task embedding predicts the task type (pick-and-place vs. stack vs. sort). This is a simple cross-entropy classification loss that encourages the embedding to encode task identity. Weight: 0.1x the main loss.

**Object identification loss:** A small MLP head predicts which object colors were involved in the demonstrated task (e.g., "red" and "blue"). Multi-label binary cross-entropy. This encourages the embedding to encode which objects matter. Weight: 0.1x the main loss.

These auxiliary heads are discarded after training — they're only used to shape the task embedding during learning.

### 8.3 Training Hyperparameters

- Batch size: 64 (demonstration, timestep) pairs
- Learning rate: 1e-4 with cosine decay to 1e-6 over training
- Optimizer: AdamW with weight decay 1e-4
- Training epochs Phase 1: 100-200 epochs over the full dataset
- Vision encoder: frozen (no gradients) — only the projection layers, temporal transformer, and policy transformer are trained
- Total trainable parameters: ~15-25M (small — this is not a huge model)
- Estimated Phase 1 training time: 8-12 hours on a single A100
- Estimated Phase 2 (DAgger iterations): 4-6 hours per iteration, 3-5 iterations = 12-30 hours total

### 8.4 Compute Budget

| Phase | Hours on A100 | Estimated Cost (Lambda ~$1.50/hr) |
|---|---|---|
| Phase 1 training | 10 | $15 |
| Phase 2 DAgger (5 rounds) | 25 | $37 |
| LLM fine-tuning (LoRA) | 5 | $8 |
| Experimentation / debugging | 10 | $15 |
| **Total** | **~50** | **~$75** |

---

## 9. Evaluation

### 9.1 Metrics

**Primary metric — Task Success Rate:** Did the robot complete the demonstrated task? Binary per episode. Measured as:

- For pick-and-place: Is the target object within 2cm of the goal position?
- For stacking: Is object A resting stably on top of object B?

**Secondary metrics:**

- **Efficiency:** Number of timesteps to complete the task (fewer = better)
- **Generalization gap:** Success rate on training scenes minus success rate on held-out scenes. Smaller gap = better generalization
- **One-shot vs. few-shot:** Success rate when given 1 demo vs. 3 demos vs. 5 demos of the same task

### 9.2 Evaluation Protocol

- 50 held-out scene configurations per task type (never seen during training)
- 10 held-out demonstrations per task type (recorded separately from training demos)
- Each evaluation episode: randomly pair one held-out demo with one held-out scene
- Total evaluation: 500 episodes per task type (50 scenes × 10 demos)
- Report mean success rate with standard error

### 9.3 Baselines to Compare Against

To make the project rigorous, compare EchoTask against:

1. **Blind replay:** Simply replay the exact joint trajectory from the demonstration in the new scene. This will fail badly (objects are in different positions) but establishes a floor.
2. **No demonstration (random policy):** Robot takes random actions. Success rate should be near 0%. Confirms the task is non-trivial.
3. **State-based oracle:** Policy receives ground truth object positions instead of camera images. This is the ceiling — how well could you do with perfect perception?
4. **EchoTask without temporal encoder:** Replace the demonstration transformer with simple frame averaging. Shows the value of temporal understanding.

### 9.4 Target Performance

Reasonable targets based on existing one-shot imitation learning literature:

| Task | Training Scenes | Held-out Scenes |
|---|---|---|
| Pick and Place | 85-95% | 65-80% |
| Stack | 80-90% | 55-70% |
| Sort | 70-85% | 45-60% |

If you hit these numbers, the project is a clear success. If held-out performance on pick-and-place exceeds 70% from a single demonstration, that's a strong result.

---

## 10. Demo Production Plan

The demo is how people experience EchoTask. It matters as much as the model.

### 10.1 Demo Format

A 60-90 second screen recording with the following structure:

**0-10s:** Title card. "EchoTask: One-Shot Imitation Learning." Clean, minimal.

**10-30s:** Show the demonstration. A human teleoperates the robot to pick up a red block and stack it on a blue one. The scene is clearly visible. Text overlay: "Human demonstrates the task once."

**30-35s:** Scene transition. Objects rearrange to a completely new configuration. Text overlay: "New scene. Different layout. Robot has never seen this arrangement."

**35-55s:** The robot executes the task autonomously. Camera captures it in real time. The robot approaches the red block, picks it up, moves it to the blue block, places it on top. Text overlay: "One demonstration. Zero re-programming."

**55-70s:** Quick montage of 3-4 additional scenes, each different, the robot succeeding each time. Shows generalization is real, not a cherry-picked example.

**70-90s:** LLM narration overlay on the final execution. The model describes what it's doing in real time as text on screen: "Identified red block at position (0.15, -0.08). Planning approach trajectory. Grasping. Lifting. Moving to blue block. Placing. Task complete." End card with GitHub link.

### 10.2 Recording Setup

- Use robosuite's built-in rendering at 256x256 or 512x512 for the demo video
- Record at 30fps for smooth video (model runs at 10Hz internally, interpolate for visual smoothness)
- Two camera angles: the model's actual input camera + a "cinematic" side angle for visual appeal
- Compose the split-screen and overlays in any video editor (DaVinci Resolve is free)

---

## 11. Project Milestones

### Milestone 1 — "I Can See" (End of Week 2)

- MuJoCo + robosuite installed and running on M1 Pro
- Franka Panda robot arm rendering with a table and colored cubes
- Camera view displaying correctly
- Keyboard teleoperation working — you can manually control the robot and complete a pick-and-place task
- Recording pipeline saves demonstrations to HDF5 files
- **Deliverable:** A screen recording of you teleoperating the robot to stack a block

### Milestone 2 — "I Have Data" (End of Week 4)

- Scripted expert policy generates demonstrations automatically for pick-and-place and stacking
- 500+ demonstrations collected per task type across randomized scenes
- Data loading pipeline reads HDF5 files, applies augmentations, produces training batches
- Vision encoder (SigLIP or DINOv2) runs on a sample image and produces feature vectors
- **Deliverable:** A visualization showing 20 random demonstrations side by side, demonstrating scene diversity

### Milestone 3 — "It Learns on Training Scenes" (End of Week 7)

- Full architecture (demonstration encoder + scene perceiver + policy) implemented
- Phase 1 behavioral cloning training completed on cloud GPU
- Policy achieves >80% success rate on training scenes for pick-and-place
- **Deliverable:** Video of the robot succeeding on 10 consecutive training scenes

### Milestone 4 — "It Generalizes" (End of Week 10)

- DAgger refinement completed (3-5 rounds)
- Policy achieves >65% success rate on held-out scenes for pick-and-place from a single demonstration
- Evaluation protocol run, metrics computed, baselines compared
- **Deliverable:** Evaluation results table and video of the robot succeeding on held-out scenes

### Milestone 5 — "Holy Shit Demo" (End of Week 12)

- LLM fine-tuned for task narration
- Demo video produced with split-screen format
- GitHub repo cleaned up with README, architecture diagram, and results
- Blog post or write-up explaining the project
- **Deliverable:** The 90-second demo video and a polished GitHub repository

---

## 12. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Generalization to new scenes is very poor (<30%) | Medium | High | Increase scene randomization, add more demonstrations, try few-shot (3-5 demos) instead of strict one-shot |
| Behavioral cloning compounds errors badly | High | Medium | DAgger is specifically designed to fix this — budget time for it |
| Vision encoder features don't distinguish objects well enough | Low | High | Try multiple encoders (SigLIP vs. DINOv2 vs. CLIP). Fine-tune the last 2 layers if frozen doesn't work |
| MuJoCo rendering is too slow on M1 Pro for data collection | Low | Medium | Reduce image resolution to 84x84 during collection, upscale for training |
| 50 hours of cloud GPU isn't enough | Medium | Low | Budget is conservative. Can extend to 80 hours (~$120) if needed |
| Teleoperation is too tedious for 1000+ demos | High | Medium | Lean heavily on scripted expert. Only teleoperate ~100-200 demos manually |
| The LLM narration feels gimmicky | Low | Low | It's a demo bonus, not core functionality. Cut it if it doesn't add value |

---

## 13. Repository Structure

```
EchoTask/
├── README.md
├── DESIGN_DOC.md (this document)
├── requirements.txt
├── configs/
│   ├── env_config.yaml         # simulation parameters
│   ├── model_config.yaml       # architecture hyperparameters
│   └── train_config.yaml       # training hyperparameters
├── echotask/
│   ├── env/
│   │   ├── setup.py            # environment initialization
│   │   ├── teleop.py           # teleoperation interface
│   │   └── scripted_expert.py  # automated demonstration generator
│   ├── data/
│   │   ├── collector.py        # demonstration recording
│   │   ├── dataset.py          # PyTorch dataset class
│   │   └── augmentations.py    # image augmentation pipeline
│   ├── models/
│   │   ├── demo_encoder.py     # demonstration → task embedding
│   │   ├── scene_perceiver.py  # current image → scene features
│   │   ├── policy.py           # task embedding + scene → action
│   │   └── llm_narrator.py     # LLM-based task narration
│   ├── training/
│   │   ├── bc_trainer.py       # behavioral cloning
│   │   ├── dagger.py           # DAgger refinement loop
│   │   └── llm_finetune.py     # LoRA fine-tuning for narrator
│   └── evaluation/
│       ├── evaluator.py        # run evaluation episodes
│       ├── metrics.py          # compute success rates
│       └── baselines.py        # baseline comparisons
├── scripts/
│   ├── collect_demos.py        # run data collection
│   ├── train.py                # run training
│   ├── evaluate.py             # run evaluation
│   └── record_demo_video.py    # produce the demo video
├── demos/
│   └── (recorded HDF5 files)
└── results/
    └── (evaluation logs and videos)
```

---

## 14. Dependencies

- Python 3.10+
- MuJoCo >= 3.0 (free, open source)
- robosuite >= 1.4
- PyTorch >= 2.0
- torchvision
- transformers (Hugging Face — for vision encoder and LLM)
- peft (Hugging Face — for LoRA)
- h5py (for demonstration storage)
- numpy, scipy
- imageio (for video recording)
- pyyaml (for config files)
- wandb (optional — for experiment tracking)