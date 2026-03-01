# Phase 1: Simulation Foundations

**Milestone:** M1 — "I Can See" (first half)
**Prerequisites:** Python 3.10+, macOS with Apple Silicon (M1 Pro or later)
**Deliverables:** MuJoCo + robosuite running, robot rendering on screen, keyboard teleoperation working, first successful pick-and-place via manual control

---

## Overview

Before any learning can happen, you need a world for the robot to live in. This phase sets up the simulation environment — the physics engine, the robot arm, the objects on the table, and the camera that sees it all. By the end, you'll control a 7-DOF robot arm with your keyboard and complete a pick-and-place task manually.

This phase is pure infrastructure. No neural networks, no data, no training. Just a working simulation you can see and interact with.

---

## Concepts

### What Is a Physics Simulator?

A physics simulator computes what happens when objects interact — gravity pulls things down, collisions stop objects from passing through each other, friction prevents sliding. In robotics, the simulator replaces the real world during development. You can crash the robot 10,000 times with zero cost.

**MuJoCo** (Multi-Joint dynamics with Contact) is the physics engine EchoTask uses. It handles:

- **Rigid body dynamics:** How solid objects move, rotate, and collide
- **Contact physics:** What happens when the gripper touches a block — friction, normal forces, slipping
- **Joint dynamics:** How motor torques at each robot joint translate into arm movement
- **Numerical integration:** Stepping the simulation forward in time (default: 500 Hz internal physics, rendered at lower rates)

MuJoCo is fast and accurate. It's the standard engine for manipulation research (used by DeepMind, OpenAI, and most academic robotics labs). Since version 2.1 it's open source and free.

### What Is Robosuite?

Robosuite is a framework built on top of MuJoCo that saves you from writing low-level simulation code. Without it, you'd need to:

- Write XML model files describing every link and joint of the robot
- Implement control algorithms from scratch
- Build your own rendering pipeline
- Create task environments manually

Robosuite provides all of this pre-built:

- **Robot models:** Franka Panda (and others) with tuned dynamics parameters
- **Task environments:** Pre-defined manipulation tasks with object spawning and success detection
- **Controllers:** Multiple control modes including Operational Space Control (OSC)
- **Rendering:** Camera image generation for training vision models
- **Teleoperation:** Built-in keyboard and SpaceMouse control interfaces

### The Franka Emika Panda

The Panda is a 7-degree-of-freedom (7-DOF) robot arm — it has 7 joints, each of which can rotate independently. Think of your own arm: shoulder (3 DOF), elbow (1 DOF), wrist (3 DOF) = 7 DOF. The Panda has the same kinematic structure, roughly.

**Why 7 DOF?** Six degrees of freedom are needed to place the end-effector (the gripper) at any position and orientation in 3D space (3 for position, 3 for orientation). The 7th degree of freedom gives redundancy — there are multiple joint configurations that reach the same gripper pose, which helps avoid joint limits and singularities.

The gripper is a **parallel-jaw gripper** — two flat fingers that move symmetrically toward or away from each other. It can grasp objects by squeezing them between the fingers. Simple but effective for blocks and cylinders.

### Operational Space Control (OSC)

You don't want to think about 7 individual joint angles when controlling the robot. You want to say "move the gripper 2cm to the right." **Operational Space Control** does this translation.

**How OSC works conceptually:**

1. You specify a desired Cartesian movement: (dx, dy, dz) — move the gripper by this much in world coordinates
2. The controller computes the Jacobian — a matrix that maps between joint velocities and end-effector velocities
3. It solves for the joint torques that will produce the desired Cartesian motion
4. These torques are sent to the simulated motors

The key benefit: **the policy's action space matches the controller's input space.** The neural network will output (dx, dy, dz, gripper) and OSC handles the joint-level math. This is critical — if the policy had to output 7 joint torques directly, the learning problem would be much harder.

### Camera Rendering

The camera is the robot's eye. In simulation, a camera is defined by:

- **Position:** Where in 3D space the camera sits
- **Orientation:** Which direction it points (look-at point + up vector)
- **Field of view:** How wide the camera's vision cone is
- **Resolution:** How many pixels the image has (128x128 for training, 256x256 for demo)

MuJoCo renders images by computing what each pixel "sees" — tracing rays from the camera through the scene, computing lighting, shadows, and material colors. This gives you an RGB image that looks like what a real camera would see.

**EchoTask camera setup:** One fixed camera mounted above and behind the robot, angled downward at ~45 degrees. This gives a good overhead view of the table workspace. A second cinematic camera (side angle) is only used for demo video recording, not as model input.

### Teleoperation

Teleoperation means a human controls the robot remotely in real time. In EchoTask, this is how you'll create the first demonstrations — you physically guide the robot through a task using your keyboard.

**Robosuite keyboard mapping:**
- Arrow keys → XY movement on the table plane
- Page Up / Page Down → Z movement (up/down)
- Spacebar → Toggle gripper open/close

During teleoperation, every timestep records: RGB image, end-effector position (x,y,z), gripper state, and joint positions. This data becomes a demonstration.

---

## Implementation Guide

### Step 1: Install MuJoCo

MuJoCo 3.x is installed via pip — no manual downloads needed:

```bash
pip install mujoco
```

Verify the installation:

```python
import mujoco
print(mujoco.__version__)  # Should print 3.x.x
```

On Apple Silicon, MuJoCo uses the Metal rendering backend. If you see OpenGL errors, ensure you're running natively (not under Rosetta).

### Step 2: Install Robosuite

```bash
pip install robosuite
```

Verify by loading a test environment:

```python
import robosuite as suite
env = suite.make(
    "Lift",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)
env.reset()
env.render()
```

If a window appears showing the Panda arm and a table with a cube, the installation works.

### Step 3: Configure the EchoTask Environment

Create a custom environment configuration in `configs/env_config.yaml`. Key parameters:

- Robot: Panda
- Controller: OSC_POSE (Operational Space Control)
- Table size and position
- Object definitions: cubes (3cm), cylinders (3cm diameter, 5cm height)
- Object colors: 8-color palette for randomization
- Camera: position, orientation, resolution (128x128 training / 256x256 demo)

Write the environment setup in `backend/env/setup.py` (per CLAUDE.md, backend files go in the backend folder). This file should:

1. Initialize robosuite with the correct controller configuration
2. Spawn the table and objects with randomized positions and colors
3. Configure the camera view
4. Return the environment ready for interaction

### Step 4: Set Up Camera Rendering

Configure the offscreen renderer for capturing training images:

```python
env = suite.make(
    "Lift",
    robots="Panda",
    has_renderer=False,           # No on-screen window
    has_offscreen_renderer=True,  # Render to image array
    use_camera_obs=True,          # Include camera images in observations
    camera_names="agentview",     # Camera name
    camera_heights=128,           # Training resolution
    camera_widths=128,
)
```

The `obs` dictionary returned by `env.step()` will include an `"agentview_image"` key containing a 128x128x3 numpy array (RGB image).

### Step 5: Implement Keyboard Teleoperation

Robosuite provides a teleoperation utility. Wire it up in `backend/env/teleop.py`:

1. Create the environment with `has_renderer=True` (on-screen rendering for visual feedback)
2. Use robosuite's `KeyboardDevice` or implement a simple keyboard listener
3. Map keyboard inputs to OSC action vectors: arrow keys → (dx, dy), PgUp/PgDn → dz, Space → gripper toggle
4. At each step, convert key presses to a 4D action vector: (dx, dy, dz, gripper)
5. Call `env.step(action)` and render

### Step 6: Test Manual Task Completion

Launch the teleoperation interface and try to:

1. Move the gripper above a cube
2. Lower the gripper
3. Close the gripper to grasp the cube
4. Lift the cube
5. Move it to a target location
6. Open the gripper to release

This should be achievable in 15-30 seconds. If it feels sluggish or unresponsive, adjust the action scaling (how much each keypress moves the robot).

---

## Learning Checkpoints

**Conceptual checks:**

- [ ] Can you explain what MuJoCo handles vs. what robosuite handles?
- [ ] Why does EchoTask use OSC instead of direct joint control?
- [ ] What are the 7 degrees of freedom, and why is the 7th useful?
- [ ] What resolution are training images vs. demo images, and why the difference?

**Practical milestones:**

- [ ] MuJoCo + robosuite installed and verified
- [ ] Custom environment loads with Panda arm, table, and colored objects
- [ ] Camera renders 128x128 RGB images of the scene
- [ ] Keyboard teleoperation moves the robot smoothly in all 3 axes + gripper
- [ ] You can complete a pick-and-place task manually via teleoperation
- [ ] Screen recording of a manual block-stacking task captured

---

## Key Files

| File | Purpose |
|------|---------|
| `configs/env_config.yaml` | Simulation parameters (robot, objects, camera, randomization) |
| `backend/env/setup.py` | Environment initialization and configuration |
| `backend/env/teleop.py` | Keyboard teleoperation interface |

---

## Common Pitfalls

1. **Rendering failures on macOS:** MuJoCo 3.x on Apple Silicon requires native ARM execution. If running under Rosetta (x86 emulation), rendering may crash. Ensure your Python environment is ARM-native.

2. **Action scaling too large or small:** If each keypress moves the robot 10cm, control will be jerky and imprecise. If it moves 0.1mm, it'll feel frozen. Start with ~1cm per keypress and adjust.

3. **Forgetting offscreen renderer:** If you only set `has_renderer=True` (on-screen), you won't get camera observation arrays. For data collection, you need `has_offscreen_renderer=True` and `use_camera_obs=True`.

4. **Gripper force tuning:** If the gripper can't hold objects (blocks slip out), the grip force may need increasing in the controller config. Robosuite's default Panda gripper settings should work for small blocks, but verify.

5. **OSC controller mode confusion:** Robosuite offers `OSC_POSE` (position + orientation control) and `OSC_POSITION` (position only). For EchoTask, `OSC_POSE` is recommended — even though the policy only outputs position deltas, having orientation control available prevents the gripper from tilting unexpectedly.

---

## Further Reading

- [MuJoCo Documentation](https://mujoco.readthedocs.io/) — Official docs, especially the modeling guide for understanding XML scene descriptions
- [Robosuite Documentation](https://robosuite.ai/) — Environment setup, controller details, and task definitions
- [Franka Emika Panda Datasheet](https://franka.de/research) — Real robot specifications (helps understand the simulation model)
- [Operational Space Control (Khatib 1987)](https://cs.stanford.edu/group/manips/publications/pdfs/Khatib_1987_pedestrians.pdf) — The foundational paper on OSC for robot manipulation
- **Next phase:** [Phase 2: Data Pipeline](phase-02-data-pipeline.md) — Recording demonstrations and building the training dataset
