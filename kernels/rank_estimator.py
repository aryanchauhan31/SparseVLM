import torch
def sketch_rank(
    A: torch.Tensor,
    n_iter: int = 4,
    oversample: int = 10,
) -> torch.Tensor:
  
    *batch_dims, M, N = A.shape
    device = A.device
    dtype  = A.dtype

    small_dim = min(M, N)
    if small_dim <= 200:
        k = small_dim
    else:
        k = min(small_dim, int(small_dim ** 0.5) + oversample)

    A_flat = A.reshape(-1, M, N)
    B_size = A_flat.shape[0]

    compute_dtype = torch.float32 if dtype == torch.bfloat16 else dtype
    A_compute = A_flat.to(compute_dtype)

    Omega = torch.randn(B_size, N, k, device=device, dtype=compute_dtype)
    Y = torch.bmm(A_compute, Omega)                            

    for _ in range(n_iter):
        Y = torch.bmm(A_compute, torch.bmm(A_compute.transpose(1, 2), Y))

    Q, _ = torch.linalg.qr(Y)                              
    B_proj = torch.bmm(Q.transpose(1, 2), A_compute)       
    _, S, _ = torch.linalg.svd(B_proj, full_matrices=False) 

    thresh = S.amax(dim=-1, keepdim=True) * 1e-5
    ranks  = (S > thresh).sum(dim=-1)

    return ranks.reshape(*batch_dims)


def estimate_prune_counts(
    P: torch.Tensor,
    n_vis_tokens: int,
) -> torch.Tensor:
    ranks = sketch_rank(P)
    prune_counts = (0.5 * (n_vis_tokens - ranks)).int()
    return prune_counts.clamp(min=0, max=n_vis_tokens - 1)
