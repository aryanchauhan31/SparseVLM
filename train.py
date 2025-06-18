import torch
from torch.utils.data import DataLoader
import deepspeed
from transformers import AutoTokenizer
import mpi4py
from deepspeed.ops.adam import DeepSpeedCPUAdam

# === Load ===
dataset = COCODataset(
    image_dir="train2017",
    annotation_file="annotations/captions_train2017.json"
)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = SparseVLMModel()
optimizer = DeepSpeedCPUAdam(model.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)


# === DeepSpeed Config (inline) ===
ds_config = {
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu"
        },
        "offload_param": {
            "device": "cpu"
        },
        "overlap_comm": True,
        "contiguous_gradients": True
    },
    "gradient_clipping": 1.0,
    "steps_per_print": 10,
    "wall_clock_breakdown": False
}

# === Init DeepSpeed ===
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=None,
    model=model,
    optimizer=optimizer,
    config=ds_config
)

# === Train ===
model_engine.train()
for epoch in range(5):
    for step, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(model_engine.device)
        input_ids = batch["input_ids"].to(model_engine.device)
        labels = batch["labels"].to(model_engine.device)

        outputs = model_engine(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
        loss = outputs.loss

        model_engine.backward(loss)
        model_engine.step()

        if step % 5 == 0:
            print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")

# After training (outside the loop)
model_engine.save_pretrained("sparsevlm_ckpt")

