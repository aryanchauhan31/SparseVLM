import torch
import torch.nn as nn
from transformers import CLIPVisionModel, LlamaForCausalLM

class SparseVLMModel(nn.Module):
  def __init__(self, clip_model='openai/clip-vit-base-patch16', llama_model='meta-llama/Llama-2-7b-hf', top_k=32):
    super().__init__()
    self.vision_encoder = CLIPVisionModel.from_pretrained(clip_model)
    self.language_model = LlamaForCausalLM.from_pretrained(llama_model)
    self.vision_proj = nn.Linear(
        self.vision_encoder.config.hidden_size,
        self.language_model.config.hidden_size
    )
    self.token_scorer = nn.Sequential(
        nn.LayerNorm(self.vision_encoder.config.hidden_size),
        nn.Linear(self.vision_encoder.config.hidden_size, 1)
    )
    self.top_k = top_k

  def forward(self, pixel_values, input_ids, labels=None):
    vision_outputs = self.vision_encoder(pixel_values=pixel_values)
    vision_tokens = vision_outputs.last_hidden_state
    patch_tokens = vision_tokens[:, 1:, :]

    scores = self.token_scorer(patch_tokens).squeeze(-1)
    topk = torch.topk(scores, self.top_k, dim=1).indices

    batch_size = patch_tokens.size(0)
    topk_tokens = torch.stack([
        patch_tokens[i, topk[i]] for i in range(batch_size)
    ])
    
    visual_embeds = self.vision_proj(topk_tokens)
    input_embeds = self.language_model.embed_tokens(input_ids)
    combined_embeds = torch.cat([visual_embeds, input_embeds], dim=1)
    
    outputs = self.language_model(inputs_embeds=combined_embeds, labels=labels)
    return outputs





