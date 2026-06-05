import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        ],
        key=["K", "N_text", "D"],
    )
    @triton.jit
    def _sparse_attn_kernel(
        Q_ptr, K_ptr, Out_ptr,
        stride_qb, stride_qk, stride_qd,
        stride_kb, stride_kn, stride_kd,
        stride_ob, stride_ok, stride_on,
        B: tl.constexpr,
        K: tl.constexpr,
        N_text: tl.constexpr,
        D: tl.constexpr,
        scale,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)

        Q_base = Q_ptr + pid_b * stride_qb
        q_mask = (offs_m[:, None] < K) & (offs_d[None, :] < D)
        q = tl.load(
            Q_base + offs_m[:, None] * stride_qk + offs_d[None, :] * stride_qd,
            mask=q_mask, other=0.0,
        )

        K_base = K_ptr + pid_b * stride_kb
        k_mask = (offs_n[:, None] < N_text) & (offs_d[None, :] < D)
        k = tl.load(
            K_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=k_mask, other=0.0,
        )

        scores = tl.dot(q, tl.trans(k)) * scale

        Out_base = Out_ptr + pid_b * stride_ob
        out_mask = (offs_m[:, None] < K) & (offs_n[None, :] < N_text)
        tl.store(
            Out_base + offs_m[:, None] * stride_ok + offs_n[None, :] * stride_on,
            scores, mask=out_mask,
        )


def _sparse_attn_triton(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    B, Kk, D = Q.shape
    _, N_text, _ = K.shape
    scale = D ** -0.5
    Out = torch.empty(B, Kk, N_text, device=Q.device, dtype=Q.dtype)

    def grid(meta):
        return (
            triton.cdiv(Kk, meta["BLOCK_M"]),
            triton.cdiv(N_text, meta["BLOCK_N"]),
            B,
        )

    _sparse_attn_kernel[grid](
        Q, K, Out,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        B=B, K=Kk, N_text=N_text, D=D, scale=scale,
    )
    return Out


def _sparse_attn_pytorch(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    scale = Q.shape[-1] ** -0.5
    return torch.bmm(Q, K.transpose(1, 2)) * scale


def sparse_vision_attn(
    patch_tokens: torch.Tensor,    
    text_embeds: torch.Tensor,     
    kept_indices: torch.Tensor,     
    use_triton: bool = True,
) -> torch.Tensor:                  

    B, N_vis, D = patch_tokens.shape
    _, K = kept_indices.shape

    idx = kept_indices.unsqueeze(-1).expand(B, K, D)
    Q = torch.gather(patch_tokens, dim=1, index=idx).contiguous()
    K_mat = text_embeds.contiguous()

    if use_triton and TRITON_AVAILABLE and Q.is_cuda:
        return _sparse_attn_triton(Q, K_mat)
    return _sparse_attn_pytorch(Q, K_mat)
