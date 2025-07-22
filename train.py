import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import tempfile

def main(rank=0, world_size=1):
    # ==== Distributed init ====
    temp_dir = tempfile.mkdtemp()
    init_method = f"file://{temp_dir}/sharedfile"
    dist.init_process_group(
        backend='nccl',
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    # ==== Dataset & Loader ====
    dataset = COCODataset(
        image_dir="train2017",
        annotation_file="annotations/captions_train2017.json"
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=4)

    # ==== Model & Optimizer ====
    torch.cuda.set_device(rank)
    model = SparseVLMModel().cuda(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)  # <-- robust for unused params
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # ==== Training Loop ====
    for epoch in range(1):
        model.train()
        sampler.set_epoch(epoch)
        epoch_loss = 0.0

        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=(rank != 0))):
            pixel_values = batch["pixel_values"].cuda(rank, non_blocking=True)
            input_ids = batch["input_ids"].cuda(rank, non_blocking=True)
            labels = batch["labels"].cuda(rank, non_blocking=True)

            outputs = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        # Aggregate loss across processes for printing
        total_loss = torch.tensor(epoch_loss, device=rank)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        avg_loss = total_loss.item() / world_size / len(dataloader)
        if rank == 0:
            print(f"[Epoch {epoch+1}] Average Loss: {avg_loss:.4f}")

        # Save checkpoint from rank 0 only
        if rank == 0:
            save_dir = f"./sparsevlm_checkpoint_epoch{epoch+1}"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.module.state_dict(), os.path.join(save_dir, "model.pt"))

    dist.destroy_process_group()

if __name__ == "__main__":
    # For multi-GPU: use mp.spawn OR torchrun
    # world_size = torch.cuda.device_count()
    # mp.spawn(main, args=(world_size,), nprocs=world_size)

    # For single-GPU (Jupyter/Notebook), debug:
    main(rank=0, world_size=1)
