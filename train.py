import torch
from torch.utils.data import DataLoader
import deepspeed
from transformers import AutoTokenizer
import mpi4py
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import get_scheduler
from tqdm import tqdm

# Optional: wandb logging
use_wandb = False
if use_wandb:
    import wandb
    wandb.init(project="sparsevlm-coco")

# === Dataset + Loader ===
dataset = COCODataset(
    image_dir="train2017",
    annotation_file="/content/annotations/captions_train2017.json"
)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# === Model & Optimizer ===
model = SparseVLMModel()
optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(model.parameters(), lr=5e-5)

# === DeepSpeed Config ===
ds_config = {
    "train_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
      }
    },
    "optimizer": {
        "type": "DeepSpeedCPUAdam",
        "params": {
            "lr": 5e-6,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "lr_scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 5e-5,
            "warmup_num_steps": 100
        }
    },
    "gradient_clipping": 1.0
}


# === DeepSpeed Initialize ===
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=ds_config
)


# === Training Loop ===
for epoch in range(5):
    model_engine.train()
    epoch_loss = 0.0

    for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
        pixel_values = batch["pixel_values"].to(model_engine.device)
        input_ids = batch["input_ids"].to(model_engine.device)
        labels = batch["labels"].to(model_engine.device)

        outputs = model_engine(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels
        )
        loss = outputs.loss
        model_engine.backward(loss)
        torch.nn.utils.clip_grad_norm_(model_engine.parameters(), max_norm=1.0)
        model_engine.step()

        epoch_loss += loss.item()
        if use_wandb:
            wandb.log({"train/loss": loss.item(), "step": step + epoch * len(dataloader)})

    avg_loss = epoch_loss / len(dataloader)
    print(f"[Epoch {epoch+1}] Average Loss: {avg_loss:.4f}")

    # === Save checkpoint ===
    save_dir = f"./sparsevlm_checkpoint_epoch{epoch+1}"
    model_engine.save_checkpoint(save_dir)

