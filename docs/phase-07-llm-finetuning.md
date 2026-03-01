# Phase 7: LLM Fine-tuning

**Milestone:** M5 — "Holy Shit Demo" (first half)
**Prerequisites:** [Phase 6: Generalization](phase-06-generalization.md) complete — model generalizing to held-out scenes
**Deliverables:** Fine-tuned LLM that produces structured task descriptions from demonstration frames

---

## Overview

EchoTask's core pipeline (demonstration encoder → scene perceiver → policy) doesn't use language at all. The LLM is an auxiliary component that serves two purposes:

1. **Training-time:** Generate structured task labels that improve the task embedding through auxiliary losses
2. **Demo-time:** Narrate the robot's actions in real-time for the final video

This phase teaches you how Large Language Models work, how to fine-tune one efficiently with LoRA, and how to build a multi-modal pipeline that takes image sequences as input and produces structured text as output.

---

## Concepts

### What Is a Large Language Model?

An LLM is a Transformer neural network trained to predict the next token (word piece) in a sequence. The core mechanism:

```
Input:  "The robot picked up the red"
Output: "block" (predicted next token, highest probability)
```

Through predicting next tokens on trillions of words of text, the model learns:

- Language structure (grammar, syntax)
- World knowledge (objects, physics, common sense)
- Reasoning patterns (if A then B, cause and effect)
- Instruction following (when trained on instruction-response pairs)

**Scale matters.** A 125M parameter model can complete sentences. A 8B parameter model can follow complex instructions, reason about images, and produce structured outputs. EchoTask uses an 8B model because it needs to understand visual scenes and produce precise structured descriptions.

### The Transformer Architecture in LLMs

LLMs use the **decoder-only Transformer** — the same architecture as GPT. Key difference from the ViT encoder you used in Phase 3:

**Encoder Transformer (ViT):** Every token can attend to every other token (bidirectional). Good for understanding an entire input.

**Decoder Transformer (LLM):** Each token can only attend to previous tokens (causal/autoregressive). Good for generating text left-to-right.

**Why causal masking?** Generation must be sequential — when predicting the 5th word, the model can only see words 1-4. If it could see word 6, it would be "cheating" by looking at the future.

**The generation process:**

```
1. Input: "Task type:"
2. Model predicts next token: "pick" (highest probability)
3. Input becomes: "Task type: pick"
4. Model predicts: "_and"
5. Input becomes: "Task type: pick_and"
6. Model predicts: "_place"
7. ... continues until end-of-sequence token
```

### Llama 3 8B — The Base Model

EchoTask uses **Llama 3 8B** as the base LLM. Key specs:

| Property | Value |
|----------|-------|
| Parameters | 8 billion |
| Architecture | Decoder-only Transformer |
| Layers | 32 |
| Hidden dimension | 4096 |
| Attention heads | 32 |
| Context length | 8192 tokens |
| Vocabulary | 128K tokens (BPE tokenizer) |
| Memory (fp16) | ~16 GB |
| Memory (quantized int8) | ~8 GB |

**Why Llama 3 8B?** It's the sweet spot for EchoTask — large enough to understand visual scenes and produce structured output, small enough to fine-tune on a single A100 in a few hours. If memory is tight, Phi-3 Mini (3.8B) is a fallback.

### Multi-Modal Input — Images + Text

Llama 3 8B is a text-only model. To feed it demonstration images, you need a **multi-modal adapter** — a module that converts image features into token-like embeddings the LLM can process.

**How it works:**

```
Demo frames (N images)
    ↓ Frozen vision encoder (SigLIP/DINOv2)
N × 768-dim feature vectors
    ↓ Linear projection (learnable)
N × 4096-dim "visual tokens"
    ↓ Prepend to text token embeddings
[visual_token_1] [visual_token_2] ... [visual_token_N] [text tokens...]
    ↓ LLM processes the full sequence
Structured output
```

The visual tokens are treated exactly like text tokens by the LLM — they go through the same attention layers. The LLM learns to "read" the visual tokens during fine-tuning, extracting information about objects, actions, and spatial relationships.

**Key design choice:** Use a small number of visual tokens (1 per frame, using the CLS feature). With 30 demo frames, that's 30 visual tokens — well within the 8192-token context window, leaving plenty of room for the text output.

### LoRA — Efficient Fine-tuning

Fine-tuning all 8B parameters is:
- **Expensive:** Requires storing optimizer states for 8B parameters (~48 GB for AdamW)
- **Risky:** Can destroy the model's general capabilities ("catastrophic forgetting")
- **Unnecessary:** You only need to adapt a small fraction of the model's behavior

**LoRA (Low-Rank Adaptation)** solves this by freezing the original model and injecting small trainable matrices into the attention layers.

**How LoRA works mathematically:**

In a standard linear layer: `y = Wx` where W is a (d_out × d_in) matrix.

LoRA adds a low-rank update: `y = Wx + BAx` where:
- A is a (rank × d_in) matrix — projects input to low rank
- B is a (d_out × rank) matrix — projects back to output dimension
- rank is small (16 in EchoTask, vs. d_in and d_out which are 4096)

**Why this works:** The adaptation needed for a specific task typically lives in a low-dimensional subspace. You don't need to modify all 8B parameters — modifying a low-rank subspace of the attention weights is sufficient.

**LoRA hyperparameters:**

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| Rank (r) | 16 | Dimension of the low-rank matrices. Higher = more capacity, more parameters |
| Alpha (α) | 32 | Scaling factor. The LoRA update is scaled by α/r. Alpha=2r is a common setting |
| Target modules | q_proj, v_proj | Which attention matrices get LoRA adapters. Query and Value projections are most important |
| Dropout | 0.05 | Regularization on LoRA layers |

**Trainable parameters with LoRA:**
- Per attention layer: 2 × rank × hidden_dim × 2 (for q and v projections) = 2 × 16 × 4096 × 2 = 262,144
- 32 layers: ~8.4M trainable parameters
- That's **0.1%** of the total 8B parameters

**The math behind alpha:** The LoRA update is `ΔW = BA × (α/r)`. With r=16 and α=32, the scaling factor is 2.0. This means the LoRA update is amplified 2x relative to its raw magnitude. Higher alpha = more aggressive adaptation. The α/r ratio controls how much the fine-tuning changes the model's behavior.

### Structured Output — What the LLM Produces

The LLM's task is to convert visual demonstrations into structured task descriptions:

**Input (prompt):**
```
You are analyzing a robot demonstration. Given the following sequence
of visual observations from a robotic manipulation task, produce a
structured task description.

[visual_token_1] [visual_token_2] ... [visual_token_30]

Output the task description in the following format:
```

**Expected output:**
```json
{
  "task_type": "pick_and_place",
  "source_object": "red block",
  "target": "on top of blue block",
  "waypoints": [
    "approach red block",
    "grasp",
    "lift",
    "move to blue block",
    "release"
  ]
}
```

This structured output is used in two ways:
1. **Training auxiliary signal:** The task classification and object identification labels come from these descriptions
2. **Demo narration:** The text descriptions are displayed during the live demo

### Training Data for the LLM

You need ~5,000 (image sequence, structured label) pairs to fine-tune the LLM.

**How to bootstrap the training data:**

1. **Generate with a larger model:** Use Claude or GPT-4V to label a seed set of ~500 demonstrations. Show the model a grid of demo frames and ask for the structured description.
2. **Scripted labels:** For demonstrations from the scripted expert, you already know the task type, source object, and target (since you programmed the expert). Generate labels programmatically.
3. **Manual cleanup:** Review ~200 labels for quality, fix errors, establish consistency.
4. **Expand:** Use the programmatic labels for the bulk of the data, manual/model labels for diversity.

---

## Implementation Guide

### Step 1: Set Up the LLM Environment

Install the required packages:

```bash
pip install transformers peft accelerate bitsandbytes
```

Load Llama 3 8B with quantization (to fit in A100 memory alongside EchoTask):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # 8-bit quantization, ~8GB instead of ~16GB
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    quantization_config=quantization_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
```

### Step 2: Build the Multi-Modal Adapter

Create a visual token projection in `backend/models/llm_narrator.py`:

```python
class VisualProjection(nn.Module):
    def __init__(self, visual_dim=768, llm_dim=4096):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(visual_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, visual_features):
        # visual_features: (B, N_frames, 768)
        return self.projection(visual_features)  # (B, N_frames, 4096)
```

This projection maps each frame's CLS feature (768-dim) into the LLM's token embedding space (4096-dim).

### Step 3: Apply LoRA

Using Hugging Face PEFT:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                           # Rank
    lora_alpha=32,                  # Alpha
    target_modules=["q_proj", "v_proj"],  # Which layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 8,388,608 || all params: 8,030,261,248 || trainable%: 0.1044%
```

### Step 4: Prepare Training Data

Create the training dataset with (visual_features, prompt, target_output) triples:

1. For each demonstration, extract visual features using the frozen encoder (same one EchoTask uses)
2. Construct the prompt template with visual token placeholders
3. Format the target output as the structured JSON
4. Tokenize the combined prompt + target for causal LM training

```python
class LLMFineTuneDataset(Dataset):
    def __init__(self, demos, labels, vision_encoder, tokenizer):
        self.demos = demos
        self.labels = labels
        self.vision_encoder = vision_encoder
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        demo = self.demos[idx]
        label = self.labels[idx]

        # Extract visual features
        with torch.no_grad():
            visual_features = self.vision_encoder.extract_global(demo["frames"])

        # Format text
        prompt = self.format_prompt()
        target = json.dumps(label, indent=2)
        full_text = prompt + target

        # Tokenize
        tokens = self.tokenizer(full_text, return_tensors="pt", truncation=True)

        return {
            "visual_features": visual_features,
            "input_ids": tokens["input_ids"],
            "labels": tokens["input_ids"],  # Causal LM: predict next token
        }
```

### Step 5: Fine-tune

Create `backend/training/llm_finetune.py`:

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="checkpoints/llm_narrator",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 16
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
```

**Training time:** ~4-6 hours on a single A100 for 5,000 examples, 3 epochs.

### Step 6: Evaluate the LLM

Test the fine-tuned LLM on held-out demonstrations:

1. Feed visual features + prompt, let the model generate
2. Parse the JSON output
3. Verify: correct task type? correct object identification? correct waypoints?
4. Compute accuracy per field

Expected performance: >90% accuracy on task type, >85% on object identification, >70% on complete structured output match.

---

## Learning Checkpoints

**Conceptual checks:**

- [ ] What's the difference between an encoder-only Transformer (ViT) and a decoder-only Transformer (LLM)?
- [ ] Why does LoRA freeze the original weights and add low-rank matrices instead of fine-tuning everything?
- [ ] What does "rank 16" mean in LoRA? What would happen with rank 1 vs. rank 128?
- [ ] How does the alpha/rank ratio affect fine-tuning? What happens if alpha is too high?
- [ ] Why use visual tokens (embedding projection) instead of describing the images in text?
- [ ] Why is the LLM not used during real-time robot execution?

**Practical milestones:**

- [ ] Llama 3 8B loaded with 8-bit quantization
- [ ] LoRA applied — verify only ~8.4M parameters are trainable
- [ ] Visual projection maps 768-dim features to 4096-dim LLM space
- [ ] Training data prepared: 5,000 (visual_features, structured_label) pairs
- [ ] Fine-tuning completes in ~4-6 hours on A100
- [ ] LLM produces valid structured JSON from unseen demonstrations
- [ ] >90% accuracy on task type identification

---

## Key Files

| File | Purpose |
|------|---------|
| `backend/models/llm_narrator.py` | Visual projection + LLM wrapper for narration |
| `backend/training/llm_finetune.py` | LoRA fine-tuning script |
| `configs/llm_config.yaml` | LLM hyperparameters (rank, alpha, learning rate) |
| `scripts/finetune_llm.py` | CLI script to launch LLM fine-tuning |

---

## Common Pitfalls

1. **Out of memory during LLM fine-tuning:** Even with 8-bit quantization and LoRA, the LLM uses ~12-14GB. Combined with EchoTask's frozen encoder (~350MB), you need ~15GB. On an A100 (40GB), this is fine. On a T4 (16GB), it's tight. Use `gradient_checkpointing=True` to trade compute for memory.

2. **JSON parsing failures in LLM output:** The LLM may produce malformed JSON (missing quotes, extra commas). Add a post-processing step that attempts to fix common JSON errors, or constrain generation with a structured output library.

3. **LoRA applied to wrong modules:** If you apply LoRA to `k_proj` and `o_proj` instead of `q_proj` and `v_proj`, it still works but less effectively. The query and value projections are the highest-impact targets for task adaptation.

4. **Visual tokens not properly integrated:** The visual token embeddings must be inserted into the right position in the input sequence (before the text tokens) and the attention mask must include them. If the attention mask treats visual tokens as padding, the LLM ignores them.

5. **Catastrophic forgetting with too many epochs:** 3 epochs is usually sufficient for LoRA fine-tuning on 5K examples. More epochs may cause the model to overfit to the training prompt format and fail on slight variations.

6. **Tokenizer padding side:** Llama models use left-padding by default. If you accidentally right-pad, the causal mask is wrong and the model can't attend to the visual tokens properly.

---

## Further Reading

- [LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971) — The original Llama paper, explains the architecture
- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685) — The LoRA paper, read sections 1-4 for theory and method
- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft) — Practical guide to applying LoRA with the PEFT library
- [Visual Instruction Tuning (LLaVA, Liu et al., 2023)](https://arxiv.org/abs/2304.08485) — The seminal paper on visual token projection for LLMs, very relevant to EchoTask's approach
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — Visual explanation of decoder-only Transformer generation
- **Previous phase:** [Phase 6: Generalization](phase-06-generalization.md)
- **Next phase:** [Phase 8: Inference & Demo](phase-08-inference-demo.md) — Building the live inference pipeline and demo video
