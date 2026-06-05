import torch
import torch.nn as nn
from kernels.token_scorer import (
    select_raters, score_visual_tokens,
    compute_prune_counts, get_kept_and_deleted_indices,
    recycle_and_cluster,
)


def default_target_layers(n_layers):
    return [i for i in range(2, n_layers, 4)]


def _get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if (hasattr(model, "model") and hasattr(model.model, "language_model")
            and hasattr(model.model.language_model, "layers")):
        return model.model.language_model.layers
    raise ValueError(
        f"Cannot find decoder layers in {type(model).__name__}. "
        "Tried model.model.layers and model.model.language_model.layers."
    )

def _make_pre_hook(shared_state, is_target=False):
    def pre_hook(module, args, kwargs):
        pid = shared_state.get("position_ids")
        pe = shared_state.get("position_embeddings")
        am = shared_state.get("attention_mask")
        need_update = pid is not None or pe is not None or am is not None or is_target
        if not need_update:
            return args, kwargs
        kwargs = dict(kwargs)
        if pid is not None:
            kwargs["position_ids"] = pid
        if pe is not None:
            kwargs["position_embeddings"] = pe
        if am is not None:
            kwargs["attention_mask"] = am
        if is_target:
            kwargs["output_attentions"] = True
        return args, kwargs
    return pre_hook


def _make_post_hook(shared_state, layer_idx, min_keep, tau, theta):
    def post_hook(module, args, kwargs, output):
        n_vis = shared_state["n_vis"]
        if n_vis <= min_keep:
            return output

        hidden_check = output[0]
        if hidden_check.shape[1] <= 1:
            return output

        hidden_out = output[0]
        rest = list(output[1:])


        attn_weights = None
        attn_rest_idx = None
        for i, r in enumerate(rest):
            if r is not None and torch.is_tensor(r) and r.dim() == 4:
                attn_weights = r
                attn_rest_idx = i
                break

        if attn_weights is None:
            return output   

        B, H, N_total, _ = attn_weights.shape
        device = hidden_out.device

        A_tv = attn_weights[:, :, n_vis:, :n_vis].mean(dim=1)

        rater_mask = select_raters(A_tv)
        n_raters = rater_mask.sum(dim=-1)
        vision_scores, A_rater = score_visual_tokens(A_tv, rater_mask)
        prune_counts = compute_prune_counts(
            A_rater.float(), n_raters, n_vis, min_keep
        )
        kept_list, deleted_list, deleted_scores_list = \
            get_kept_and_deleted_indices(vision_scores, prune_counts)

        vis_tokens = hidden_out[:, :n_vis, :]
        text_tokens = hidden_out[:, n_vis:, :]
        new_seqs = []
        new_n_vis_list = []

        for b in range(B):
            kept     = vis_tokens[b, kept_list[b]]
            recycled = None
            if deleted_list[b].numel() > 0:
                recycled = recycle_and_cluster(
                    vis_tokens[b, deleted_list[b]],
                    deleted_scores_list[b],
                    tau=tau, theta=theta,
                )
            parts = [kept]
            if recycled is not None:
                parts.append(recycled)
            parts.append(text_tokens[b])
            new_seqs.append(torch.cat(parts, dim=0))
            new_n_vis_list.append(
                kept.shape[0] + (recycled.shape[0] if recycled is not None else 0)
            )

        max_len = max(s.shape[0] for s in new_seqs)
        D = hidden_out.shape[-1]
        padded = torch.zeros(B, max_len, D, device=device, dtype=hidden_out.dtype)
        for b, seq in enumerate(new_seqs):
            padded[b, :seq.shape[0]] = seq

        new_n_vis  = min(new_n_vis_list)
        hidden_out = padded
        shared_state["n_vis"] = new_n_vis

        n_text  = text_tokens.shape[1]
        kept0   = kept_list[0].to(device)          
        text_ix = torch.arange(n_vis, n_vis + n_text, device=device)
        kept_all = torch.cat([kept0, text_ix])


        pid = shared_state.get("position_ids")
        if pid is not None:
            shared_state["position_ids"] = (
                pid[:, kept_all] if pid.dim() == 2 else pid[:, :, kept_all]
            )


        pe = shared_state.get("position_embeddings")
        if pe is not None:
            cos, sin = pe
            shared_state["position_embeddings"] = (
                cos[:, kept_all, :], sin[:, kept_all, :]
            )

        am = shared_state.get("attention_mask")
        if am is not None and am.dim() == 4:
            shared_state["attention_mask"] = \
                am[:, :, kept_all, :][:, :, :, kept_all]

        # Remove attn_weights from output (caller didn't request them)
        if attn_rest_idx is not None:
            rest[attn_rest_idx] = None

        return (hidden_out,) + tuple(rest)

    return post_hook


def patch_qwen2vl(model, n_vis, target_layers=None,
                  min_keep=32, tau=0.5, theta=0.5):
    layers        = _get_layers(model)
    n_layers      = len(layers)
    target_layers = target_layers or default_target_layers(n_layers)
    target_set    = set(target_layers)

    shared_state = {
        "n_vis": n_vis,
        "position_ids": None,
        "position_embeddings": None,
        "attention_mask": None,
        "_hooks": [],
    }

    for layer_idx, layer in enumerate(layers):
        is_target = layer_idx in target_set

        h_pre = layer.register_forward_pre_hook(
            _make_pre_hook(shared_state, is_target=is_target), with_kwargs=True
        )
        shared_state["_hooks"].append(h_pre)

        if is_target:
            h_post = layer.register_forward_hook(
                _make_post_hook(shared_state, layer_idx, min_keep, tau, theta),
                with_kwargs=True,
            )
            shared_state["_hooks"].append(h_post)

    n_pre    = n_layers
    n_target = len(target_set)
    print(
        f"[SparseVLM] Registered hooks on {n_pre} layers "
        f"(pre-hook all, post-hook at {sorted(target_set)}). "
        f"n_vis={n_vis}, min_keep={min_keep}."
    )
    return shared_state


def reset_n_vis(shared_state, n_vis):
    shared_state["n_vis"] = n_vis
    shared_state["position_ids"] = None
    shared_state["position_embeddings"] = None
    shared_state["attention_mask"] = None


def unpatch_qwen2vl(model):
    print("[SparseVLM] unpatch: use the state dict's '_hooks' list to remove hooks.")
    print("  Hint: for h in state['_hooks']: h.remove()")


def remove_hooks(shared_state):

    for h in shared_state.get("_hooks", []):
        h.remove()
    shared_state["_hooks"] = []
    print(f"[SparseVLM] All hooks removed.")
