import torch
import torch.nn as nn
from transformers import CLIPVisionModel, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence

class SparseVLMModel(nn.Module):
    def __init__(self, clip_model='openai/clip-vit-base-patch16', llama_model='EleutherAI/gpt-neo-125M', top_k=32):
        super().__init__()
        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_model)
        self.language_model = AutoModelForCausalLM.from_pretrained(llama_model)

        self.vision_hidden_size = getattr(self.vision_encoder.config, "hidden_size", 768)
        self.text_hidden_size = getattr(self.language_model.config, "hidden_size", 768)

        self.vision_proj = nn.Linear(self.vision_hidden_size, self.text_hidden_size)
        self.vis_ln = nn.LayerNorm(self.text_hidden_size)
        self.txt_ln = nn.LayerNorm(self.text_hidden_size)

        self.top_k = top_k

    def recycle_and_cluster(self, deleted_tokens, deleted_scores, tau=0.5, theta=0.5):
        if deleted_tokens.size(0) < 1:
            return None

        num_recycle = int(tau * deleted_tokens.size(0))
        if num_recycle < 1:
            return None

        recycle_idx = torch.topk(deleted_scores, num_recycle).indices
        recycled_tokens = deleted_tokens[recycle_idx]

        num_clusters = max(1, int(theta * num_recycle))
        recycled_norm = torch.nn.functional.normalize(recycled_tokens, dim=-1)

        centers = [recycled_norm[0]]
        for _ in range(1, num_clusters):
            dists = torch.stack([
                1 - torch.matmul(recycled_norm, c.unsqueeze(-1)).squeeze(-1)
                for c in centers
            ], dim=1).min(dim=1).values
            next_center = recycled_norm[dists.argmax()]
            centers.append(next_center)

        assignments = torch.stack([
            torch.matmul(recycled_norm, c.unsqueeze(-1)).squeeze(-1)
            for c in centers
        ], dim=1).argmax(dim=1)

        aggregated = []
        for k in range(num_clusters):
            members = recycled_tokens[assignments == k]
            if members.size(0) > 0:
                aggregated.append(members.sum(dim=0))

        return torch.stack(aggregated) if aggregated else None

    def forward(self, pixel_values, input_ids, labels=None):
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        vision_tokens = vision_outputs.last_hidden_state
        patch_tokens = vision_tokens[:, 1:, :]  # drop CLS token

        if hasattr(self.language_model, "model") and hasattr(self.language_model.model, "embed_tokens"):
            embed_tokens = self.language_model.model.embed_tokens
        elif hasattr(self.language_model, "transformer") and hasattr(self.language_model.transformer, "wte"):
            embed_tokens = self.language_model.transformer.wte
        else:
            raise ValueError("Cannot find token embedding layer in language model.")

        text_embeds = embed_tokens(input_ids)
        patch_tokens = patch_tokens.to(dtype=text_embeds.dtype)

        # Sparse attention logic
        Attn_scores = torch.matmul(patch_tokens, text_embeds.transpose(1, 2))
        Attn_softmax = torch.softmax(Attn_scores, dim=-1).float()
        text_relevance_scores = Attn_softmax.mean(dim=1)

        mean_relevance = text_relevance_scores.mean(dim=1, keepdim=True)
        raters_mask = text_relevance_scores >= mean_relevance
        raters_mask = raters_mask.unsqueeze(1).expand(-1, patch_tokens.size(1), -1)

        Attn_softmax = Attn_softmax * raters_mask.float()
        vision_scores = (Attn_softmax.sum(dim=-1) / raters_mask.sum(dim=-1).clamp(min=1e-5)).float()

        P = Attn_softmax.transpose(1, 2).float()
        ranks = torch.stack([torch.linalg.matrix_rank(P[i]) for i in range(P.size(0))])
        prune_counts = (0.5 * (patch_tokens.size(1) - ranks)).int()

        topk_masked_tokens = []
        for i in range(patch_tokens.size(0)):
            N = prune_counts[i].item()
            K = patch_tokens.size(1) - N

            scores = vision_scores[i]
            topk_idx = torch.topk(scores, K).indices

            all_idx = torch.arange(patch_tokens.size(1), device=scores.device)
            deleted_mask = torch.ones_like(all_idx, dtype=torch.bool)
            deleted_mask[topk_idx] = False
            deleted_idx = all_idx[deleted_mask]

            deleted_tokens = patch_tokens[i, deleted_idx]
            deleted_scores = scores[deleted_idx].float()

            aggregated_tokens = self.recycle_and_cluster(deleted_tokens, deleted_scores)
            selected_tokens = patch_tokens[i, topk_idx]
            if aggregated_tokens is not None:
                selected_tokens = torch.cat([selected_tokens, aggregated_tokens], dim=0)

            topk_masked_tokens.append(selected_tokens)

        topk_tokens = pad_sequence(topk_masked_tokens, batch_first=True)

        # Match dtype
        visual_embeds = self.vis_ln(self.vision_proj(topk_tokens)).to(dtype=text_embeds.dtype)
        input_embeds = self.txt_ln(text_embeds)

        combined_embeds = torch.cat([visual_embeds, input_embeds], dim=1)

        # Pad labels
        if labels is not None:
            visual_pad = torch.full(
                (labels.size(0), visual_embeds.size(1)),
                fill_value=-100,
                dtype=labels.dtype,
                device=labels.device
            )
            padded_labels = torch.cat([visual_pad, labels], dim=1)
        else:
            padded_labels = None

        return self.language_model(inputs_embeds=combined_embeds, labels=padded_labels)
