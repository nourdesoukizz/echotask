# Phase 6: Generalization — DAgger, Evaluation, and Ablations

**Milestone:** M4 — "It Generalizes"
**Prerequisites:** [Phase 5: Training](phase-05-training.md) complete — behavioral cloning model achieving >80% on training scenes
**Deliverables:** DAgger refinement running, >65% success on held-out scenes, full evaluation protocol with baselines and ablations

---

## Overview

Your Phase 5 model works on training scenes but struggles on new ones. This is the fundamental problem of behavioral cloning: **distribution shift.** The model has only seen states that an expert would visit, so when it makes a small mistake and ends up in an unfamiliar state, errors compound rapidly.

This phase fixes that with DAgger (Dataset Aggregation), then establishes rigorous evaluation with baselines and ablation studies. By the end, EchoTask generalizes to scenes it has never trained on.

---

## Concepts

### Distribution Shift — Why Behavioral Cloning Fails

Behavioral cloning trains on expert demonstrations where every state is one the expert visited. But at test time, the policy makes its own decisions, and small mistakes push it into states the expert never visited.

**The compounding error problem:**

```
Timestep 1: Policy is slightly off → 1mm positioning error
Timestep 5: Error accumulates → 5mm off the expert trajectory
Timestep 10: Now in a state never seen during training → policy is confused
Timestep 15: Confused policy makes a large error → task fails
```

**Analogy:** Imagine learning to ride a bike by watching videos. The videos show the bike perfectly balanced. But when you actually ride, you wobble. The videos never showed "what to do when wobbling left" because the expert never wobbled. You fall.

The mathematical framing: If the policy has error ε per timestep, over T timesteps the total error grows as O(ε × T²) — **quadratic** in the horizon. For a 200-timestep episode, even a tiny per-step error becomes catastrophic.

### DAgger (Dataset Aggregation)

DAgger solves distribution shift by training the policy on states it actually visits, not just states the expert visited.

**The DAgger algorithm:**

```
1. Initialize dataset D₀ with expert demonstrations (from Phase 2)
2. Train policy π₁ on D₀ (this is your Phase 5 model)
3. For round i = 1, 2, ..., K:
   a. Roll out policy πᵢ in simulation to collect trajectories
      - The policy controls the robot, visiting its own states
   b. At each state the policy visits, query the expert for the
      correct action (what SHOULD the robot do from here?)
   c. Add these (state, expert_action) pairs to the dataset: Dᵢ = Dᵢ₋₁ ∪ new_data
   d. Retrain the policy on the expanded dataset Dᵢ → πᵢ₊₁
4. Return the final policy πₖ₊₁
```

**Why DAgger works:** After round 1, the dataset contains states the policy actually visits — including the "mistake" states. The expert labels tell the model "when you're 5mm off target, here's how to correct." The policy learns to recover from its own errors.

**Querying the expert in EchoTask:** You use the scripted expert from Phase 2. When the learned policy is at position (x, y, z) and needs to pick up the red block at (bx, by, bz), the scripted expert computes the optimal action: move toward (bx, by, bz). This works because the scripted expert has access to ground truth object positions from the simulator.

**How many DAgger rounds?** 3-5 rounds is typical. Diminishing returns after that — each round adds less new information because the policy's state distribution stabilizes.

**DAgger round cost:**
- Each round: run the policy for ~500 episodes to collect data, then retrain
- ~500 episodes × ~200 timesteps × ~0.1s/step = ~2.5 hours of simulation per round
- Plus ~4-6 hours of retraining per round
- Total for 5 rounds: ~30-40 hours of compute

### Evaluation Protocol

Rigorous evaluation is what separates a research project from a demo hack. You need to answer: "How well does this actually work, and is the improvement from each component real?"

**EchoTask evaluation setup:**

- **50 held-out scenes** per task type (object positions never seen during training)
- **10 held-out demonstrations** per task type (separate from training demos)
- **500 evaluation episodes** per task type: 50 scenes × 10 demos, randomly paired
- Report **mean success rate ± standard error**

**Why 500 episodes?** Statistical significance. With a 65% success rate and 500 trials, the standard error is ~2%. This means you can confidently distinguish 65% from 60% performance. With only 50 episodes, the standard error would be ~7%, making it hard to tell if improvements are real.

### Baselines — What Are You Comparing Against?

Baselines establish the floor and ceiling of performance, letting you understand where EchoTask sits:

#### 1. Blind Replay (Floor)

Simply replay the exact joint trajectory from the demonstration in the new scene. Objects are in different positions, so this fails badly — the robot reaches for where the block was in the demo scene, not where it is now.

**Expected success rate:** ~0-5% (might accidentally succeed if objects happen to be in similar positions).

**Why include it:** Proves that the task requires adaptation, not just memorization.

#### 2. Random Policy (Sub-floor)

The robot takes random actions at each timestep.

**Expected success rate:** ~0% (near impossible to randomly complete a pick-and-place).

**Why include it:** Confirms the task is non-trivial. If random succeeds at 5%, your task is too easy.

#### 3. State-Based Oracle (Ceiling)

Replace camera images with **ground truth object positions** from the simulator. The policy receives (object_x, object_y, object_z) directly instead of pixel features.

**Expected success rate:** 90-99% (perfect perception, only policy errors remain).

**Why include it:** Shows how much performance is limited by perception vs. policy. If EchoTask gets 65% and the oracle gets 95%, the 30% gap is due to imperfect visual perception — that's your bottleneck.

#### 4. No Temporal Encoder (Ablation)

Replace the Transformer-based demonstration encoder with simple **mean pooling** — average all frame features into a single vector, ignoring temporal order.

**Expected success rate:** 10-20% lower than full EchoTask.

**Why include it:** Quantifies the value of temporal understanding. If this baseline performs similarly to EchoTask, the temporal Transformer isn't contributing much and you should investigate why.

### Ablation Studies

Ablations isolate the contribution of each design decision:

| Ablation | What Changes | Tests |
|----------|-------------|-------|
| No auxiliary losses | Remove task classification + object ID losses | Do auxiliary losses help the task embedding? |
| No data augmentation | Remove all image augmentations during training | How much does augmentation improve generalization? |
| No DAgger | Use Phase 5 model directly (BC only) | How much does DAgger improve robustness? |
| Fewer demonstrations | Train with 100 demos instead of 500 | How data-efficient is the approach? |
| DINOv2 instead of SigLIP | Swap vision encoder | Which encoder produces better features for this task? |

Run each ablation on the full evaluation protocol (500 episodes). Report the success rate difference relative to the full model. A difference of >5% with low standard error is meaningful.

---

## Implementation Guide

### Step 1: Implement the DAgger Loop

Create `backend/training/dagger.py`:

```python
class DAggerTrainer:
    def __init__(self, model, env, scripted_expert, bc_trainer, config):
        self.model = model
        self.env = env
        self.expert = scripted_expert
        self.bc_trainer = bc_trainer
        self.num_rounds = config.dagger_rounds  # 3-5
        self.episodes_per_round = config.dagger_episodes  # ~500
        self.beta_schedule = np.linspace(1.0, 0.0, self.num_rounds)  # Expert mixing

    def run(self):
        for round_idx in range(self.num_rounds):
            beta = self.beta_schedule[round_idx]

            # 1. Collect rollouts using the current policy
            new_data = self.collect_rollouts(beta)

            # 2. Add new data to the training dataset
            self.bc_trainer.dataset.add_data(new_data)

            # 3. Retrain the model on the expanded dataset
            self.bc_trainer.train(num_epochs=50)  # Fewer epochs per round

            # 4. Evaluate
            success_rate = self.evaluate()
            print(f"Round {round_idx}: beta={beta:.2f}, success={success_rate:.2%}")

    def collect_rollouts(self, beta):
        """Roll out the policy, query expert at each state."""
        all_data = []
        for ep in range(self.episodes_per_round):
            obs = self.env.reset()
            demo = self.get_random_demo()
            task_emb = self.model.encode_demonstration(demo)

            for t in range(max_steps):
                # With probability beta, use expert action (for stability early on)
                expert_action = self.expert.get_action(self.env)

                if random.random() < beta:
                    action = expert_action
                else:
                    action = self.model.predict(task_emb, obs)

                # Record the state and the EXPERT'S action (not the policy's)
                all_data.append({
                    "image": obs["image"],
                    "robot_state": obs["robot_state"],
                    "expert_action": expert_action,
                    "demo": demo,
                })

                obs, _, done, _ = self.env.step(action)
                if done:
                    break

        return all_data
```

**The beta schedule:** In early rounds, mix in expert actions (beta=1.0 → mostly expert) to prevent the policy from diverging into completely unrecoverable states. Gradually decrease beta so the policy visits its own states. By the final round, beta=0.0 and the policy acts entirely on its own.

### Step 2: Implement the Evaluation Pipeline

Create `backend/evaluation/evaluator.py`:

```python
class Evaluator:
    def __init__(self, model, env, held_out_demos, held_out_scenes):
        self.model = model
        self.env = env
        self.demos = held_out_demos      # 10 demos per task
        self.scenes = held_out_scenes    # 50 scene configs per task

    def evaluate(self, num_episodes=500):
        successes = []
        episode_lengths = []

        for i in range(num_episodes):
            demo = self.demos[i % len(self.demos)]
            scene = self.scenes[i % len(self.scenes)]

            success, length = self.run_episode(demo, scene)
            successes.append(success)
            episode_lengths.append(length)

        success_rate = np.mean(successes)
        std_error = np.std(successes) / np.sqrt(len(successes))

        return {
            "success_rate": success_rate,
            "std_error": std_error,
            "mean_episode_length": np.mean(episode_lengths),
            "num_episodes": num_episodes,
        }

    def run_episode(self, demo, scene_config, max_steps=300):
        obs = self.env.reset_to_scene(scene_config)
        task_emb = self.model.encode_demonstration(demo)

        for t in range(max_steps):
            action = self.model.predict(task_emb, obs)
            obs, reward, done, info = self.env.step(action)
            if done:
                return info["success"], t + 1

        return False, max_steps  # Timed out
```

### Step 3: Implement Baselines

Create `backend/evaluation/baselines.py`:

1. **BlindReplayBaseline:** Load joint trajectory from demo, replay it in the new scene
2. **RandomPolicyBaseline:** Output uniform random actions in the valid range
3. **StateBasedOracleBaseline:** Replace vision features with ground truth object positions, train a separate small policy MLP
4. **NoTemporalEncoderBaseline:** Replace the demo encoder with mean pooling of frame features

### Step 4: Implement Metrics

Create `backend/evaluation/metrics.py`:

```python
def compute_metrics(eval_results):
    return {
        "success_rate": np.mean(eval_results["successes"]),
        "std_error": sem(eval_results["successes"]),
        "mean_efficiency": np.mean(eval_results["episode_lengths"]),
        "generalization_gap": (
            eval_results["train_success_rate"] - eval_results["holdout_success_rate"]
        ),
    }
```

### Step 5: Run the Full Evaluation

Create `scripts/evaluate.py` that:

1. Loads the trained model checkpoint
2. Runs evaluation on training scenes (should be >80%)
3. Runs evaluation on held-out scenes (target: >65% for pick-and-place)
4. Runs all baselines
5. Produces a results table

Expected results format:

```
| Method                     | Training Scenes | Held-out Scenes | Gap    |
|----------------------------|-----------------|-----------------|--------|
| Random Policy              | 0.2% ± 0.2%    | 0.0% ± 0.0%    | —      |
| Blind Replay               | 3.1% ± 0.8%    | 2.8% ± 0.7%    | 0.3%   |
| EchoTask (BC only)         | 82.4% ± 1.7%   | 38.6% ± 2.2%   | 43.8%  |
| EchoTask (BC + DAgger)     | 88.2% ± 1.4%   | 67.4% ± 2.1%   | 20.8%  |
| EchoTask (no temporal enc) | 74.0% ± 1.9%   | 48.2% ± 2.2%   | 25.8%  |
| State-Based Oracle         | 96.4% ± 0.8%   | 93.2% ± 1.1%   | 3.2%   |
```

### Step 6: Run Ablation Studies

For each ablation, train a variant model and evaluate on the full protocol. This is compute-intensive — each ablation requires a full training run (~10 hours).

Prioritize ablations by information value:
1. **No DAgger** — already have this from Phase 5
2. **No auxiliary losses** — quick to test, retrain without aux loss terms
3. **No augmentation** — retrain without augmentations
4. **DINOv2 vs. SigLIP** — retrain with swapped encoder
5. **Fewer demos** — retrain on a 100-demo subset

---

## Learning Checkpoints

**Conceptual checks:**

- [ ] Explain distribution shift in your own words. Why does it affect behavioral cloning but not supervised image classification?
- [ ] Why does DAgger query the expert at states the *policy* visits, not at random states?
- [ ] What does the beta schedule in DAgger control, and why start with beta=1.0?
- [ ] Why do you need 500 evaluation episodes instead of 50? What changes statistically?
- [ ] What does a large generalization gap (training >> held-out) indicate?
- [ ] Why is the state-based oracle useful even though you'd never deploy it?

**Practical milestones:**

- [ ] DAgger loop runs: collect rollouts → query expert → expand dataset → retrain
- [ ] Performance improves after each DAgger round
- [ ] After 3-5 DAgger rounds: >65% success on held-out scenes for pick-and-place
- [ ] All 4 baselines implemented and evaluated
- [ ] Full results table with success rates and standard errors
- [ ] At least 3 ablation studies completed
- [ ] Video of the trained model succeeding on held-out scenes

---

## Key Files

| File | Purpose |
|------|---------|
| `backend/training/dagger.py` | DAgger refinement loop |
| `backend/evaluation/evaluator.py` | Evaluation episode runner |
| `backend/evaluation/metrics.py` | Success rate and generalization gap computation |
| `backend/evaluation/baselines.py` | Baseline comparisons (replay, random, oracle, no-temporal) |
| `scripts/evaluate.py` | CLI script for running full evaluation |
| `results/` | Evaluation logs, results tables, videos |

---

## Common Pitfalls

1. **DAgger expert queried with wrong state:** The scripted expert must use the current simulator state (where the robot actually is), not the demonstration state. If you accidentally query the expert for the demo's object positions, the labels will be wrong.

2. **Not resetting the optimizer between DAgger rounds:** When you retrain after expanding the dataset, the optimizer's momentum from the previous round may cause instability. Reset the optimizer (create a new one) or use a fresh learning rate warmup at each round.

3. **Evaluation on too few episodes:** With 50 episodes and 65% success, the 95% confidence interval is 65% ± 13%. You literally can't tell if 65% and 52% are different. Use 500 episodes minimum.

4. **Confusing the generalization gap direction:** The gap is (training performance) - (held-out performance). A large positive gap means overfitting. A negative gap (held-out > training) shouldn't happen and indicates a bug in your evaluation.

5. **DAgger data imbalance:** After 5 rounds, most of your dataset is DAgger-collected data, not original demonstrations. The original demos show optimal behavior; DAgger data shows correction from mistakes. If the balance shifts too far, the policy may learn to make mistakes so it can practice correcting them. Cap each round's new data at 50% of the original dataset size.

6. **Evaluating on scenes used for DAgger rollouts:** DAgger rollouts should use the training scene distribution. Held-out evaluation scenes must be separate. If they overlap, your held-out numbers are inflated.

---

## Further Reading

- [A Reduction of Imitation Learning to No-Regret Online Learning (Ross et al., 2011)](https://arxiv.org/abs/1011.0686) — The original DAgger paper. Read sections 1-3 for the theory, section 4 for the algorithm
- [DART: Noise Injection for Robust Imitation Learning (Laskey et al., 2017)](https://arxiv.org/abs/1703.09327) — An alternative to DAgger that injects noise during training demonstrations
- [On the Effectiveness of Fine-tuning vs. Meta-reinforcement Learning (Yu et al., 2020)](https://arxiv.org/abs/2003.11373) — Relevant ablation methodology for robotics
- [Evaluating Robot Manipulation Policies](https://robomimic.github.io/docs/introduction/results.html) — Robomimic's evaluation methodology, very similar to EchoTask's
- **Previous phase:** [Phase 5: Training](phase-05-training.md)
- **Next phase:** [Phase 7: LLM Fine-tuning](phase-07-llm-finetuning.md) — Adding language understanding to the system
