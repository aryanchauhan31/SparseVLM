from .patch import patch_qwen2vl, reset_n_vis, unpatch_qwen2vl, remove_hooks

def apply_sparsevlm(
    model,
    n_vis: int = 256,
    target_layers=None,
    min_keep: int = 32,
    tau: float = 0.5,
    theta: float = 0.5,
) -> dict:
    return patch_qwen2vl(
        model=model,
        n_vis=n_vis,
        target_layers=target_layers,
        min_keep=min_keep,
        tau=tau,
        theta=theta,
    )
  
__all__ = ["apply_sparsevlm", "reset_n_vis", "unpatch_qwen2vl", "remove_hooks"]
__version__ = "0.1.1"
