# Phase 8: Inference & Demo Production

**Milestone:** M5 — "Holy Shit Demo" (second half)
**Prerequisites:** [Phase 7: LLM Fine-tuning](phase-07-llm-finetuning.md) complete — LLM producing structured task descriptions
**Deliverables:** Real-time inference pipeline, closed-loop robot execution, 90-second demo video with LLM narration

---

## Overview

Everything built in Phases 1-7 comes together here. This is the **inference phase** — where the trained model runs in real-time, watching a demonstration once and then executing the task in a new scene.

You'll build the closed-loop execution pipeline, optimize it for real-time performance, integrate LLM narration, and produce the final demo video. By the end, you have a polished 90-second video that demonstrates one-shot imitation learning.

---

## Concepts

### What Is Inference?

**Training** is learning from data. **Inference** is using what was learned to make predictions on new inputs. In EchoTask:

| Phase | Input | Output | Speed Requirement |
|-------|-------|--------|-------------------|
| Training | 100K (demo, observation, action) triples | Updated model weights | Hours (offline) |
| Inference | 1 new demo + live scene | Robot actions | 10 Hz (real-time) |

Inference is fundamentally different from training:

- **No gradients:** No backward pass, no weight updates. Just forward passes.
- **No batching (usually):** Processing one scene at a time, not batches of 64.
- **Latency matters:** Each forward pass must complete in <100ms (10 Hz) or the robot's control loop stutters.
- **Memory is smaller:** No optimizer states, no gradient buffers. Just the model and its activations.

### Closed-Loop Execution

EchoTask uses **closed-loop control** — the robot observes the scene after every action and adjusts. This is the opposite of open-loop control (plan all actions upfront, execute blindly).

**The inference loop:**

```
1. Load demonstration → encode once → task_embedding

2. Reset scene with new object positions

3. Loop at 10 Hz:
   a. Capture camera image of current scene
   b. vision_encoder(image) → patch_features
   c. scene_perceiver(patch_features) → scene_features
   d. policy(task_embedding, scene_features, robot_state) → action
   e. Apply action to robot in simulation
   f. Check termination: policy outputs "done" signal OR max steps reached
   g. If not done → go to 3a
```

**Why closed-loop is critical:** The robot's actions are imprecise. Each 10ms action moves the gripper ~1-5mm, but maybe 0.5mm off from ideal. Without re-observing the scene, these errors accumulate. With closed-loop control, the robot sees where it actually is and corrects — approaching the block from the left? Next action shifts slightly right.

### Inference Performance Optimization

The 10 Hz requirement means each full cycle (capture → encode → perceive → act) must complete in under 100ms. Here's the time budget:

| Component | Typical Time (A100) | Typical Time (M1 Pro) |
|-----------|---------------------|----------------------|
| Camera capture (rendering) | ~5ms | ~15ms |
| Vision encoder forward pass | ~10ms | ~40ms |
| Scene perceiver | ~1ms | ~3ms |
| Policy forward pass | ~2ms | ~5ms |
| Environment step | ~2ms | ~5ms |
| **Total** | **~20ms** | **~68ms** |

Both platforms meet the 100ms budget. The vision encoder is the bottleneck. Optimization strategies if needed:

1. **Half-precision (fp16):** Run the entire model in float16 instead of float32. ~2x speedup, negligible quality loss.
2. **torch.compile:** PyTorch 2.0's compiler can fuse operations and optimize the computation graph. ~20-30% speedup.
3. **ONNX export:** Export the model to ONNX format for optimized inference runtimes.
4. **Batch vision encoding:** If you're recording a demo video (not real-time), batch multiple frames through the vision encoder together.

### The Demonstration Encoding — Run Once

The demonstration encoder processes the full demo video and produces the task embedding. This happens **once per episode**, not per timestep.

For a 30-frame demonstration:
- Extract visual features: 30 forward passes through the vision encoder (~300ms on M1 Pro)
- Temporal Transformer encoding: ~10ms
- Total: ~310ms

This is a one-time cost. After encoding, the task embedding (256-dim vector) is cached and reused for all ~200-300 timesteps of execution.

### LLM Narration — Non-Real-Time

The LLM narration runs **asynchronously** — it's not in the control loop. The robot doesn't wait for the LLM to generate text before moving.

**Narration pipeline:**

```
1. Before execution: Feed demo visual features to LLM → get task description
   "I observed the human pick up the red block and place it on the blue block."

2. During execution: At key moments (grasp, lift, place), log events
   "Approaching red block at position (0.15, -0.08)..."
   "Grasping... contact detected."
   "Lifting..."
   "Moving to blue block at position (-0.12, 0.20)..."
   "Placing... releasing gripper."
   "Task complete."

3. After execution: Combine narration text with video for the demo
```

The key-moment detection can be heuristic (gripper closing → "grasping", large z increase → "lifting") or learned (train a small classifier on robot state transitions).

### Demo Video Production

The demo video is the culmination of the project. Structure from the design doc:

**0-10s:** Title card
**10-30s:** Human teleoperation demonstration
**30-35s:** Scene transition to new layout
**35-55s:** Robot autonomous execution
**55-70s:** Montage of 3-4 additional scenes
**70-90s:** LLM narration overlay on final execution

#### Recording Setup

- **Model input camera:** 128x128 (what the model sees), upscaled for display
- **Cinematic camera:** 256x256 or 512x512, side angle, for visual appeal
- **Frame rate:** Render at 30fps for smooth video (model runs at 10 Hz, interpolate between actions for visual smoothness)
- **Two-camera composition:** Split-screen or picture-in-picture showing both views

#### Making It Look Good

1. **Cherry-pick successful episodes:** Run 50 evaluation episodes, select the 5 most visually compelling successes (smooth trajectories, diverse scenes)
2. **Consistent lighting:** Use the same lighting setup across all clips for visual coherence
3. **Smooth camera:** No camera jitter in the demo video (unlike training where jitter is added)
4. **Text overlays:** Clean, minimal text explaining what's happening
5. **Timing:** Speed up boring parts (moving through empty space), keep normal speed for key moments (grasping, placing)

---

## Implementation Guide

### Step 1: Build the Inference Pipeline

Create `backend/inference/pipeline.py`:

```python
class InferencePipeline:
    def __init__(self, model_checkpoint, device="cuda"):
        self.device = device
        self.model = self.load_model(model_checkpoint)
        self.model.eval()

    def load_model(self, checkpoint_path):
        model = EchoTaskModel(config)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        return model

    @torch.no_grad()
    def encode_demonstration(self, demo_frames, demo_ee_states):
        """Encode demo once. Returns cached task embedding."""
        demo_frames = demo_frames.to(self.device)
        demo_ee_states = demo_ee_states.to(self.device)
        task_embedding = self.model.encode_demonstration(demo_frames, demo_ee_states)
        return task_embedding

    @torch.no_grad()
    def predict_action(self, task_embedding, current_image, robot_state):
        """Predict action for a single timestep. Must be <100ms."""
        current_image = current_image.to(self.device)
        robot_state = robot_state.to(self.device)
        action = self.model(task_embedding, current_image, robot_state)
        return action.cpu().numpy()
```

### Step 2: Build the Execution Loop

Create `backend/inference/executor.py`:

```python
class ClosedLoopExecutor:
    def __init__(self, pipeline, env, max_steps=300):
        self.pipeline = pipeline
        self.env = env
        self.max_steps = max_steps

    def execute(self, demo, scene_config=None):
        """Execute a task given a demonstration."""
        # Encode demonstration (once)
        task_embedding = self.pipeline.encode_demonstration(
            demo["frames"], demo["ee_states"]
        )

        # Reset environment
        obs = self.env.reset(scene_config)
        trajectory = []

        for step in range(self.max_steps):
            # Get current observation
            image = self.preprocess_image(obs["agentview_image"])
            robot_state = obs["robot_state"]

            # Predict action
            action = self.pipeline.predict_action(
                task_embedding, image, robot_state
            )

            # Apply action
            obs, reward, done, info = self.env.step(action)

            # Record for video
            trajectory.append({
                "image": obs["agentview_image"],
                "action": action,
                "robot_state": robot_state,
                "step": step,
            })

            if done:
                break

        return {
            "success": info.get("success", False),
            "trajectory": trajectory,
            "num_steps": step + 1,
        }
```

### Step 3: Add Performance Profiling

Before recording the demo, verify the pipeline meets real-time requirements:

```python
import time

times = {"vision": [], "perceiver": [], "policy": [], "total": []}

for step in range(100):
    t0 = time.perf_counter()

    # Vision encoder
    t1 = time.perf_counter()
    features = vision_encoder(image)
    times["vision"].append(time.perf_counter() - t1)

    # Scene perceiver
    t2 = time.perf_counter()
    scene = scene_perceiver(features)
    times["perceiver"].append(time.perf_counter() - t2)

    # Policy
    t3 = time.perf_counter()
    action = policy(task_emb, scene, state)
    times["policy"].append(time.perf_counter() - t3)

    times["total"].append(time.perf_counter() - t0)

for k, v in times.items():
    print(f"{k}: {np.mean(v)*1000:.1f}ms avg, {np.max(v)*1000:.1f}ms max")
```

If any component exceeds budget, apply optimizations (fp16, torch.compile).

### Step 4: Integrate LLM Narration

Create the narration pipeline:

```python
class NarrationPipeline:
    def __init__(self, llm_model, vision_encoder):
        self.llm = llm_model
        self.vision_encoder = vision_encoder

    def narrate_demonstration(self, demo_frames):
        """Generate task description from demo (before execution)."""
        visual_features = self.vision_encoder.extract_global(demo_frames)
        prompt = self.build_demo_prompt(visual_features)
        description = self.llm.generate(prompt, max_new_tokens=200)
        return self.parse_structured_output(description)

    def narrate_execution(self, trajectory):
        """Generate execution narration (after execution)."""
        narration = []
        for event in self.detect_key_moments(trajectory):
            narration.append(self.format_event(event))
        return narration

    def detect_key_moments(self, trajectory):
        """Detect grasp, lift, place events from trajectory."""
        events = []
        for i, step in enumerate(trajectory):
            state = step["robot_state"]
            action = step["action"]

            # Gripper closing → approach/grasp
            if action[3] > 0.5 and (i == 0 or trajectory[i-1]["action"][3] < 0.5):
                events.append({"type": "grasp", "step": i, "position": state[:3]})

            # Large upward movement → lifting
            if action[2] > 0.03:
                events.append({"type": "lift", "step": i, "position": state[:3]})

            # Gripper opening after being closed → release/place
            if action[3] < -0.5 and trajectory[i-1]["action"][3] > 0.5:
                events.append({"type": "place", "step": i, "position": state[:3]})

        return events
```

### Step 5: Record the Demo Video

Create `scripts/record_demo_video.py`:

```python
class DemoVideoRecorder:
    def __init__(self, executor, narrator, env):
        self.executor = executor
        self.narrator = narrator
        self.env = env

    def record_full_demo(self, demo, scenes, output_path):
        """Record the complete 90-second demo."""
        frames = []

        # 1. Title card (10s at 30fps = 300 frames)
        frames.extend(self.create_title_card(300))

        # 2. Human demonstration (20s)
        frames.extend(self.record_demonstration(demo, fps=30))

        # 3. Scene transition (5s)
        frames.extend(self.create_transition(scenes[0], 150))

        # 4. Autonomous execution (20s)
        result = self.executor.execute(demo, scenes[0])
        narration = self.narrator.narrate_execution(result["trajectory"])
        frames.extend(self.render_execution(result["trajectory"], fps=30))

        # 5. Montage (15s — 3-4 more scenes)
        for scene in scenes[1:4]:
            result = self.executor.execute(demo, scene)
            frames.extend(self.render_execution_fast(result["trajectory"], fps=30))

        # 6. Narrated execution (20s)
        result = self.executor.execute(demo, scenes[4])
        narration = self.narrator.narrate_execution(result["trajectory"])
        frames.extend(self.render_with_narration(result["trajectory"], narration, fps=30))

        # Save video
        self.save_video(frames, output_path, fps=30)
```

**Rendering at higher resolution:** For the demo video, use 256x256 or 512x512 rendering:

```python
env = suite.make(
    ...,
    camera_heights=512,
    camera_widths=512,
)
```

The model still gets 128x128 input (downscaled), but the video uses the high-resolution render.

### Step 6: Compose the Final Video

Use robosuite's built-in rendering for raw footage, then compose in a video editor:

1. **Raw recording:** Save individual frames as images using `imageio`, then compile into a video
2. **Split-screen:** Place the model's-eye-view (128x128 upscaled) next to the cinematic view
3. **Text overlays:** Add explanatory text ("Human demonstrates the task once", "New scene. Robot has never seen this arrangement.")
4. **LLM narration text:** Overlay the generated narration as scrolling text during the final execution
5. **Trim and pace:** Speed up slow parts, add smooth transitions

```python
import imageio

writer = imageio.get_writer("demo_video.mp4", fps=30, codec="libx264", quality=8)
for frame in frames:
    writer.append_data(frame)
writer.close()
```

For split-screen composition:
```python
def create_split_frame(left_image, right_image, width=1024, height=512):
    """Combine two views side by side."""
    left_resized = cv2.resize(left_image, (width // 2, height))
    right_resized = cv2.resize(right_image, (width // 2, height))
    combined = np.concatenate([left_resized, right_resized], axis=1)
    return combined
```

---

## Learning Checkpoints

**Conceptual checks:**

- [ ] What is the difference between training and inference in terms of computation (gradients, batching, speed)?
- [ ] Why is closed-loop control critical for generalization? What would happen with open-loop execution?
- [ ] Why is the demonstration encoded once but the scene perceived every timestep?
- [ ] Why does the LLM narration run asynchronously rather than in the control loop?
- [ ] What is the bottleneck in the inference pipeline, and how can it be optimized?

**Practical milestones:**

- [ ] Inference pipeline runs a complete episode with closed-loop control
- [ ] Total per-step latency is under 100ms (verify with profiling)
- [ ] fp16 inference works without quality degradation
- [ ] LLM generates accurate task descriptions from demonstration frames
- [ ] Key moment detection identifies grasp/lift/place events from trajectories
- [ ] High-resolution (256x256 or 512x512) video recording works
- [ ] Split-screen video composition working
- [ ] Complete 90-second demo video produced
- [ ] Video shows successful task execution across 4-5 different scene configurations

---

## Key Files

| File | Purpose |
|------|---------|
| `backend/inference/pipeline.py` | Model loading + single-step prediction |
| `backend/inference/executor.py` | Closed-loop execution loop |
| `backend/models/llm_narrator.py` | LLM narration pipeline (from Phase 7) |
| `scripts/record_demo_video.py` | Demo video recording and composition |
| `results/` | Output videos and evaluation logs |

---

## Common Pitfalls

1. **Model in training mode during inference:** Forgetting `model.eval()` means dropout is active and batch normalization uses batch statistics instead of running statistics. Inference predictions become noisy and inconsistent.

2. **GPU-CPU transfer bottleneck:** If you move tensors between GPU and CPU every timestep (`action.cpu().numpy()`), the synchronization can add 5-10ms. Use `torch.cuda.synchronize()` before timing measurements to get accurate numbers. For the actual pipeline, the transfer is usually fine since it's just a 4-element action vector.

3. **Stale task embedding after model changes:** If you update the model checkpoint but reuse a cached task embedding from a previous model version, the embedding is incompatible. Always re-encode the demonstration after loading a new checkpoint.

4. **Video frame ordering issues:** Simulation rendering may not be synchronized with the control loop. Ensure you capture the frame *after* `env.step()` returns, not before, to show the result of the action.

5. **Demo video looks jerky at 10fps:** The model runs at 10 Hz but videos should be 30 fps for smooth viewing. Interpolate between control frames: for each pair of 10 Hz control frames, render 3 intermediate frames at the interpolated robot state.

6. **LLM narration doesn't match execution:** The LLM describes the task from the demonstration, but the execution might differ (e.g., approaching from a different angle). Frame the narration as intent ("Intending to place the red block on the blue one") rather than precise description of the motion.

7. **Cherry-picking too aggressively:** It's fine to show successful episodes in the demo, but don't hide the failure rate. Include the evaluation metrics (65% success rate) somewhere — honesty is more impressive than a fake 100%.

---

## Further Reading

- [Robosuite Rendering Documentation](https://robosuite.ai/docs/modules/renderers.html) — Camera setup, offscreen rendering, and video recording
- [PyTorch Inference Optimization](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) — fp16, torch.compile, and other optimization techniques
- [DaVinci Resolve (Free Video Editor)](https://www.blackmagicdesign.com/products/davinciresolve) — For composing the final demo video with overlays and transitions
- [Making Research Videos (Karpathy's Guide)](https://karpathy.github.io/) — Tips for producing clear, compelling research demo videos
- [Real-Time Policy Execution for Robot Manipulation](https://robotics-transformer-x.github.io/) — Reference architectures for real-time robot inference
- **Previous phase:** [Phase 7: LLM Fine-tuning](phase-07-llm-finetuning.md)
- **First phase:** [Phase 1: Simulation](phase-01-simulation.md) — Full circle back to where it started
