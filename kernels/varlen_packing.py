import torch
from typing import List, Tuple


def pack_varlen_batch(
    token_list: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:

    assert len(token_list) > 0
    device = token_list[0].device
    dtype  = token_list[0].dtype

    seqlens = torch.tensor(
        [t.shape[0] for t in token_list],
        dtype=torch.int32, device=device,
    )

    cu_seqlens = torch.zeros(len(token_list) + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = seqlens.cumsum(dim=0)

    packed = torch.cat(token_list, dim=0)
    return packed, cu_seqlens


def unpack_varlen_batch(
    packed: torch.Tensor,
    cu_seqlens: torch.Tensor,
    pad_to_max: bool = False,
):
    B = cu_seqlens.shape[0] - 1
    token_list = [
        packed[cu_seqlens[i]:cu_seqlens[i+1]]
        for i in range(B)
    ]

    if not pad_to_max:
        return token_list

    max_len = max(t.shape[0] for t in token_list)
    D = packed.shape[-1]
    out = torch.zeros(B, max_len, D, device=packed.device, dtype=packed.dtype)
    for i, t in enumerate(token_list):
        out[i, :t.shape[0]] = t
    return out


def packed_to_padded(
    packed: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = cu_seqlens.shape[0] - 1
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    max_len = max(seqlens)
    D = packed.shape[-1]
    device = packed.device
    dtype = packed.dtype

    padded = torch.zeros(B, max_len, D, device=device, dtype=dtype)
    mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)

    for i in range(B):
        L = seqlens[i]
        start = cu_seqlens[i].item()
        padded[i, :L] = packed[start:start + L]
        mask[i, :L] = True

    return padded, mask
