import torch
import torch.nn.functional as F
from .rank_estimator import sketch_rank

def select_raters(A_tv: torch.Tensor) -> torch.Tensor:
    mean_per_text = A_tv.mean(dim=-1)                        
    global_mean   = mean_per_text.mean(dim=-1, keepdim=True)   
    return mean_per_text > global_mean


def score_visual_tokens(
    A_tv: torch.Tensor,
    rater_mask: torch.Tensor,
) -> tuple:
    B, N_text, N_vis = A_tv.shape
    max_raters = rater_mask.sum(dim=-1).max().item()

    A_rater = torch.zeros(B, max_raters, N_vis, device=A_tv.device, dtype=A_tv.dtype)
    for b in range(B):
        rows = A_tv[b, rater_mask[b]]
        A_rater[b, :rows.shape[0]] = rows

    vision_scores = A_rater.sum(dim=1)                       
    return vision_scores, A_rater


def compute_prune_counts(
    A_rater: torch.Tensor,
    n_raters: torch.Tensor,
    N_vis: int,
    min_keep: int = 32,
) -> torch.Tensor:
    ranks = sketch_rank(A_rater)
    prune_counts = (0.5 * (N_vis - ranks.float())).int()
    return prune_counts.clamp(min=0, max=N_vis - min_keep)


def get_kept_and_deleted_indices(
    vision_scores: torch.Tensor,
    prune_counts: torch.Tensor,
) -> tuple:
    B, N_vis = vision_scores.shape
    kept_list = []
    deleted_list = []
    deleted_scores_list = []

    for b in range(B):
        P = prune_counts[b].item()
        K = N_vis - P
        topk_result = torch.topk(vision_scores[b], k=K)
        kept_idx = topk_result.indices

        all_idx = torch.arange(N_vis, device=vision_scores.device)
        deleted_mask = torch.ones(N_vis, dtype=torch.bool, device=vision_scores.device)
        deleted_mask[kept_idx] = False
        deleted_idx  = all_idx[deleted_mask]

        kept_list.append(kept_idx)
        deleted_list.append(deleted_idx)
        deleted_scores_list.append(vision_scores[b, deleted_idx])

    return kept_list, deleted_list, deleted_scores_list




def recycle_and_cluster(
    deleted_tokens: torch.Tensor,
    deleted_scores: torch.Tensor,
    tau: float = 0.5,
    theta: float = 0.5,
) -> torch.Tensor | None:
    P = deleted_tokens.shape[0]
    if P < 1:
        return None

    n_recycle = max(1, int(tau * P))
    recycle_idx = torch.topk(deleted_scores, n_recycle).indices
    recycled_tokens = deleted_tokens[recycle_idx]
    recycled_scores = deleted_scores[recycle_idx]

    n_clusters = max(1, int(theta * n_recycle))
    recycled_norm = F.normalize(recycled_tokens, dim=-1)

    centers = [recycled_norm[recycled_scores.argmax()]]
    for _ in range(1, n_clusters):
        sims = torch.stack([torch.matmul(recycled_norm, c.unsqueeze(-1)).squeeze(-1)
                               for c in centers], dim=1)
        dists = 1 - sims.max(dim=1).values
        centers.append(recycled_norm[dists.argmax()])

    sims = torch.stack([torch.matmul(recycled_norm, c.unsqueeze(-1)).squeeze(-1)
                               for c in centers], dim=1)
    assignments = sims.argmax(dim=1)

    aggregated = []
    for k in range(n_clusters):
        members = recycled_tokens[assignments == k]
        if members.shape[0] > 0:
            aggregated.append(members.sum(dim=0))

    return torch.stack(aggregated) if aggregated else None



def sparsevlm_score(
    attn_weights: torch.Tensor,     
    hidden_states: torch.Tensor,    
    n_vis: int,
    min_keep: int = 32,
    tau: float = 0.5,
    theta: float = 0.5,
) -> tuple:
    B, H, N_total, _ = attn_weights.shape

    A_tv = attn_weights[:, :, n_vis:, :n_vis].mean(dim=1)  
    rater_mask = select_raters(A_tv)
    n_raters = rater_mask.sum(dim=-1)
    vision_scores, A_rater = score_visual_tokens(A_tv, rater_mask)
    prune_counts = compute_prune_counts(A_rater, n_raters, n_vis, min_keep)
    kept_list, deleted_list, deleted_scores_list = get_kept_and_deleted_indices(
        vision_scores, prune_counts
    )

    vis_tokens = hidden_states[:, :n_vis, :]
    text_tokens = hidden_states[:, n_vis:, :]

    new_sequences = []
    new_n_vis_per_item = []

    for b in range(B):
        kept_tokens = vis_tokens[b, kept_list[b]]

        recycled = None
        if deleted_list[b].numel() > 0:
            recycled = recycle_and_cluster(
                vis_tokens[b, deleted_list[b]],
                deleted_scores_list[b],
                tau=tau, theta=theta,
            )

        parts = [kept_tokens]
        if recycled is not None:
            parts.append(recycled)
        parts.append(text_tokens[b])

        combined = torch.cat(parts, dim=0)
        new_sequences.append(combined)

        n_vis_b = kept_tokens.shape[0] + (recycled.shape[0] if recycled is not None else 0)
        new_n_vis_per_item.append(n_vis_b)

    max_len = max(s.shape[0] for s in new_sequences)
    D = hidden_states.shape[-1]
    padded = torch.zeros(B, max_len, D, device=hidden_states.device, dtype=hidden_states.dtype)
    for b, seq in enumerate(new_sequences):
        padded[b, :seq.shape[0]] = seq

    new_n_vis = min(new_n_vis_per_item)
    return padded, new_n_vis
