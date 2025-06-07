import torch
import torch.nn as nn
from transformers import CLIPVisionModel, LlamaForCausalLM

class SparseVLMModel(nn.Module):
  def __init__(self, clip_model='openai/clip-vit-base-patch16', llama_model='meta-llama/Llama-3.1-8B', top_k=32):
    super().__init__()
    self.vision_encoder = CLIPVisionModel.from_pretrained(clip_model)
    self.language_model = LlamaForCausalLM.from_pretrained(llama_model)
    self.vision_proj = nn.Linear(
        self.vision_encoder.config.hidden_size,
        self.language_model.config.hidden_size
    )
    self.top_k = top_k
    self.vision_hidden_size = self.vision_encoder.config_hidden_size
    self.text_hidden_size = self.language_model.config_hidden_size

  def forward(self, pixel_values, input_ids, labels=None):
    vision_outputs = self.vision_encoder(pixel_values=pixel_values)
    vision_tokens = vision_outputs.last_hidden_state
    patch_tokens = vision_tokens[:, 1:, :]

    text_embeds = self.language_model.model.embed_tokens(input_ids)

    Attn_scores = torch.matmul(patch_tokens, text_embeds.transpose(1,2))
    Attn_softmax = torch.softmax(Attn_scores, dim=-1)
    
    text_relevance_scores = Attn_softmax.mean(dim=1)

    mean_relevance = text_relevance_scores.mean(dim=1, keepdim=True)
    raters_mask = text_relevance_scores>=mean_relevance
    
    raters_mask = raters_mask.unsqueeze(1).expand(-1, patch_tokens.size(1),-1)
    Attn_softmax = Attn_softmax*raters_mask 
    vision_scores = Attn_softmax.sum(dim=-1)/raters_mask.sum(dim=-1).clamp(min=1e-5)

    P = Attn_softmax.transpose(1,2)
    ranks = torch.stack([
        torch.linalg.matrix_rank(P[i]) for i in range(P.size(0))
    ])

    lambda_factor = 0.5
    Lv = patch_tokens.size(1)
    prune_counts = (lambda_factor * (Lv-ranks)).init()
    batch_size = patch_tokens.size(0)
    
    top_masked_indices = []
    for i in range(batch_size):
      N = prune_counts[i].item()
      K = Lv - N
      scores = vision_scores[i]
      topk_idx = torch.topk(scores, K).indices
      top_masked_indices.append(topk_idx)

    topk_tokens = torch.stack([
        patch_tokens[i, top_masked_indices[i]] for i in range(batch_size)
    ])

    visual_embeds = self.vision_proj(topk_tokens)
    
    input_embeds = self.language_model.model.embed_tokens(input_ids)
    combined_embeds = torch.cat([visual_embeds, input_embeds], dim=1)

    outputs = self.language_model(inputs_embeds=combined_embeds, labels=labels)
    return outputs
