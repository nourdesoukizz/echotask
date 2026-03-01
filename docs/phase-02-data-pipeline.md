# Phase 2: Data Pipeline

**Milestone:** M1 (second half) — M2 "I Have Data"
**Prerequisites:** [Phase 1: Simulation](phase-01-simulation.md) complete — working simulation with teleoperation
**Deliverables:** Scripted expert generating demonstrations, 500+ demos per task, HDF5 storage, PyTorch Dataset with augmentations, visualization of scene diversity

---

## Overview

A learning system is only as good as its data. This phase builds the entire data pipeline — from recording demonstrations to loading augmented training batches. You'll write a scripted expert that automatically generates demonstrations (so you don't have to manually teleoperate 1000 times), store everything in HDF5 format, and build a PyTorch Dataset that feeds augmented data to the training loop.

By the end of this phase, you can generate thousands of demonstrations across randomized scenes and visualize the diversity of your dataset.

---

## Concepts

### What Is a Demonstration?

In imitation learning, a **demonstration** is a complete recording of an expert performing a task. For EchoTask, each demonstration contains:

| Data | Shape | Rate | Description |
|------|-------|------|-------------|
| RGB images | (T, 128, 128, 3) | 10 Hz | What the camera sees at each timestep |
| End-effector position | (T, 3) | 10 Hz | Gripper xyz in world coordinates |
| Gripper state | (T, 1) | 10 Hz | Open (-1) to closed (+1) |
| Joint positions | (T, 7) | 10 Hz | All 7 joint angles (for replay/debugging) |
| Actions taken | (T, 4) | 10 Hz | The (dx, dy, dz, gripper) commands issued |

A typical demonstration is 150-300 timesteps long (15-30 seconds at 10 Hz).

The key insight: **at training time, each (image, action) pair becomes one training example.** A single demonstration of 200 timesteps yields 200 training examples. This is why you need hundreds of demonstrations — even 500 demonstrations give you ~100,000 individual training examples.

### HDF5: The Storage Format

**HDF5** (Hierarchical Data Format version 5) is the standard storage format for robotics demonstrations. Think of it as a filesystem inside a file:

```
demo_0001.hdf5
├── images/          # (T, 128, 128, 3) uint8 array
├── ee_positions/    # (T, 3) float32 array
├── gripper_states/  # (T, 1) float32 array
├── joint_positions/ # (T, 7) float32 array
├── actions/         # (T, 4) float32 array
└── metadata/
    ├── task_type    # "pick_and_place" or "stack"
    ├── num_steps    # integer
    └── success      # boolean
```

**Why HDF5 over alternatives?**

- **vs. individual image files:** Loading 200 separate PNG files per demonstration is slow (200 file open/close operations). HDF5 stores everything in one file with fast random access.
- **vs. pickle/numpy:** HDF5 supports partial reads — you can load just the actions without loading all the images. Critical when your dataset is 50+ GB of images.
- **vs. video files:** Videos require decoding. HDF5 stores raw arrays with instant random access to any frame.

### Scripted Expert Policies

Manually teleoperating 1000 demonstrations would take ~16 hours. Instead, you write a **scripted expert** — a hardcoded controller that solves the task using privileged information (ground truth object positions from the simulator).

**How the scripted pick-and-place expert works:**

1. **Query simulator for object positions** — MuJoCo gives you the exact (x, y, z) of every object. This is "cheating" — the learned policy won't have this information. But it's fine for data generation.
2. **Plan a sequence of waypoints:**
   - Move above the target object: (obj_x, obj_y, obj_z + 0.10)
   - Lower to grasp height: (obj_x, obj_y, obj_z + 0.01)
   - Close gripper
   - Lift: (obj_x, obj_y, obj_z + 0.15)
   - Move above placement location: (place_x, place_y, place_z + 0.10)
   - Lower: (place_x, place_y, place_z + 0.01)
   - Open gripper
   - Retreat upward
3. **Convert waypoints to delta actions:** At each timestep, compute the vector from current position to the next waypoint, clip it to the maximum step size (0.05m), and output that as the action.
4. **Add noise for diversity:** Add small Gaussian noise (sigma ~0.005m) to each action. This creates natural-looking variation and prevents the learned policy from overfitting to perfectly straight trajectories.

**Data collection split:**
- ~80% from scripted expert (fast, automated, consistent)
- ~20% from manual teleoperation (adds human-like imperfection and variation)

### Scene Randomization

Every demonstration runs in a different scene configuration. The randomization parameters:

| Parameter | Range | Purpose |
|-----------|-------|---------|
| Object positions | Uniform within 30cm x 40cm workspace | Forces spatial generalization |
| Object colors | Sampled from 8-color palette | Forces color-invariant object recognition |
| Distractor objects | 0-3 additional objects not involved in task | Forces ignoring irrelevant objects |
| Lighting intensity | ±20% variation | Forces robustness to brightness changes |
| Camera jitter | ±2cm position, ±3° rotation | Forces robustness to viewpoint changes |

This randomization is the single most important factor for generalization. Without it, the model memorizes specific pixel patterns instead of learning task-level understanding.

### Data Augmentation

Scene randomization varies the **simulation** at data collection time. Data augmentation varies the **images** at training time. They serve different purposes:

- **Scene randomization** = real physical variation (objects actually move)
- **Data augmentation** = simulated visual variation (same image, different appearance)

Augmentations applied during training:

1. **Random color jitter** (brightness ±15%, contrast ±15%, saturation ±15%) — Makes the model robust to lighting and color rendering differences
2. **Random crop and resize** (crop 90-100% of image, resize back to 128x128) — Simulates slight camera zoom/shift variations
3. **Gaussian noise** (sigma = 0.01) — Simulates sensor noise
4. **Random horizontal flip** with action mirroring (flip the sign of delta-x) — Doubles effective dataset size. Important: when you flip the image horizontally, you must also negate the x-component of the action, otherwise the labels are wrong

### Building a PyTorch Dataset

A PyTorch `Dataset` is an object that the training loop queries for individual examples. For EchoTask, the dataset needs to:

1. **Index into demonstrations:** Given an index, determine which demonstration and which timestep within that demonstration to load
2. **Load the demonstration context:** For the demonstration encoder, load a subsampled sequence of frames from the same demonstration (e.g., every 5th frame, giving ~30-60 frames total)
3. **Load the current observation:** The camera image at the specific timestep
4. **Load the target action:** The expert's action at that timestep
5. **Apply augmentations:** Transform the images on-the-fly

The dataset returns a dictionary:
```python
{
    "demo_frames": Tensor(N, 3, 128, 128),    # N subsampled demo images
    "demo_ee_states": Tensor(N, 4),             # N end-effector states (xyz + gripper)
    "current_image": Tensor(3, 128, 128),       # Current scene image
    "current_ee_state": Tensor(4),              # Current end-effector state
    "target_action": Tensor(4),                 # Expert's action (dx, dy, dz, gripper)
    "task_type": int,                           # Task type label (for auxiliary loss)
}
```

---

## Implementation Guide

### Step 1: Write the Scripted Expert

Create `backend/env/scripted_expert.py` with:

1. A `ScriptedPickAndPlace` class that takes the environment and executes the pick-and-place waypoint sequence described above
2. A `ScriptedStack` class that extends pick-and-place with stacking logic (place on top of another object instead of on the table)
3. Both use `env.sim.data` to query ground truth object positions
4. Both add configurable Gaussian noise to actions
5. Both return a success/failure flag (did the task actually complete?)

### Step 2: Build the Data Collection Pipeline

Create `backend/data/collector.py` that:

1. Initializes the environment with a random scene configuration
2. Runs the scripted expert (or starts teleoperation mode)
3. Records all data streams (images, states, actions) into lists
4. Saves the completed demonstration to an HDF5 file using `h5py`
5. Loops for N demonstrations, randomizing the scene each time

The collector should be runnable as a script:
```bash
python scripts/collect_demos.py --task pick_and_place --num_demos 500 --output_dir data/demos/
```

### Step 3: Implement HDF5 Storage

Use `h5py` for reading and writing. Key considerations:

- **Compression:** Use `gzip` compression for image datasets (reduces file size ~3x with minimal read speed impact)
- **Chunking:** Set chunk size to (1, 128, 128, 3) for images — this allows efficient single-frame reads
- **Metadata:** Store task type, success flag, and number of timesteps as HDF5 attributes

```python
import h5py

with h5py.File("demo_0001.hdf5", "w") as f:
    f.create_dataset("images", data=images, compression="gzip", chunks=(1, 128, 128, 3))
    f.create_dataset("actions", data=actions)
    f.attrs["task_type"] = "pick_and_place"
    f.attrs["success"] = True
```

### Step 4: Build the PyTorch Dataset

Create `backend/data/dataset.py` with a `DemoDataset` class:

1. **`__init__`:** Scan the demo directory, build an index mapping global indices to (demo_file, timestep) pairs. Calculate total number of timesteps across all demonstrations.
2. **`__len__`:** Return total timestep count (this is the dataset size)
3. **`__getitem__`:** Given an index:
   - Look up which demo file and timestep this index corresponds to
   - Load the demo's subsampled frame sequence (for the demonstration encoder)
   - Load the current image and state at the specific timestep
   - Load the target action
   - Apply augmentations to images
   - Return the dictionary described above

### Step 5: Implement Augmentations

Create `backend/data/augmentations.py` with a `TrainingAugmentation` transform:

```python
import torchvision.transforms as T

class TrainingAugmentation:
    def __init__(self):
        self.color_jitter = T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, image, action=None):
        # Apply color jitter
        image = self.color_jitter(image)

        # Random crop and resize
        i, j, h, w = T.RandomResizedCrop.get_params(image, scale=(0.9, 1.0), ratio=(1.0, 1.0))
        image = T.functional.resized_crop(image, i, j, h, w, (128, 128))

        # Gaussian noise
        image = image + torch.randn_like(image) * 0.01

        # Random horizontal flip (with action mirroring)
        if random.random() > 0.5 and action is not None:
            image = T.functional.hflip(image)
            action[0] = -action[0]  # Negate delta-x

        # Normalize for vision encoder
        image = self.normalize(image)

        return image, action
```

### Step 6: Validate the Pipeline

Write a validation script that:

1. Generates 10 demonstrations with the scripted expert
2. Loads them through the Dataset class
3. Visualizes a grid of 20 random frames from different demonstrations
4. Verifies shapes and data ranges are correct
5. Measures data loading throughput (should sustain >100 examples/sec)

---

## Learning Checkpoints

**Conceptual checks:**

- [ ] Why does the scripted expert use ground truth positions, and why is that acceptable?
- [ ] What's the difference between scene randomization and data augmentation?
- [ ] Why must you flip the action's x-component when horizontally flipping an image?
- [ ] Why HDF5 over saving individual images or video files?
- [ ] How many training examples does a dataset of 500 demonstrations (200 timesteps each) yield?

**Practical milestones:**

- [ ] Scripted expert completes pick-and-place successfully >95% of the time
- [ ] Scripted expert completes stacking successfully >90% of the time
- [ ] 500+ demonstrations generated per task type
- [ ] HDF5 files written and readable, with correct shapes and compression
- [ ] PyTorch Dataset returns correctly shaped tensors with augmentations
- [ ] Visualization showing 20 diverse scenes from the dataset
- [ ] Data loader achieves >100 examples/second throughput

---

## Key Files

| File | Purpose |
|------|---------|
| `backend/env/scripted_expert.py` | Automated demonstration generator using ground truth positions |
| `backend/data/collector.py` | Orchestrates demo collection and HDF5 saving |
| `backend/data/dataset.py` | PyTorch Dataset class for training |
| `backend/data/augmentations.py` | Image augmentation pipeline |
| `scripts/collect_demos.py` | CLI script to run batch data collection |
| `data/demos/` | Directory where HDF5 demonstration files are stored |

---

## Common Pitfalls

1. **Scripted expert succeeds but recordings show failures:** The expert might complete the task, but the success detector in the environment may use stricter criteria (e.g., object must be within 1cm instead of 2cm). Align your success criteria between the expert and the recorder.

2. **Memory issues with large datasets:** Loading all images into memory at once will crash on a 16GB machine. The Dataset must load lazily — open HDF5 files on-demand and read only the frames needed for each `__getitem__` call.

3. **Augmentation applied to demo frames and current frame inconsistently:** The color jitter should be the same for all frames within a single training example (same brightness shift, same contrast shift). Otherwise, the temporal transformer sees frames that look like they're from different scenes.

4. **Forgetting to normalize images:** Vision encoders (SigLIP, DINOv2) expect ImageNet-normalized inputs. If you feed raw [0, 255] uint8 arrays, the features will be garbage. Always normalize after augmentation.

5. **HDF5 file locking with multi-worker DataLoader:** HDF5 files don't support concurrent reads from multiple processes by default. Set `swmr=True` (single-writer-multiple-reader) when opening files, or open/close files within each worker's `__getitem__` call.

6. **Not filtering failed demonstrations:** Some scripted expert runs will fail (object slips, collision issues). Always check the `success` flag and exclude failures from the training set.

---

## Further Reading

- [h5py Documentation](https://docs.h5py.org/) — HDF5 Python interface, especially chunking and compression
- [Robomimic](https://robomimic.github.io/) — A framework for robot learning from demonstrations that uses the same HDF5 format. Good reference for data pipeline design
- [PyTorch Data Loading Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) — Custom datasets, transforms, and DataLoader configuration
- [DAgger (Ross et al., 2011)](https://arxiv.org/abs/1011.0686) — The paper introducing Dataset Aggregation, which you'll use in Phase 6
- **Previous phase:** [Phase 1: Simulation](phase-01-simulation.md)
- **Next phase:** [Phase 3: Computer Vision](phase-03-computer-vision.md) — Extracting visual features from images
