import torch
from PIL import Image
from transformers import CLIPImageProcessor, LlamaTokenizer
from sparsevlm_backbone import SparseVLMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer & image processor
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Load model
model = SparseVLMModel().to(device)
model.eval()

# Dummy image (replace with actual file path)
image = Image.open("sample.jpg").convert("RGB")
pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(device)

# Dummy prompt
prompt = "Describe this image:"
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

# Forward pass
with torch.no_grad():
    outputs = model(pixel_values=pixel_values, input_ids=inputs.input_ids)

# Get prediction tokens
logits = outputs.logits[:, -1, :]  # Last token prediction
predicted_token_id = torch.argmax(logits, dim=-1)
generated_text = tokenizer.decode(predicted_token_id)

print("Generated output:", generated_text)
