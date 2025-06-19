import os
import json
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor, AutoTokenizer

class COCODataset(Dataset):
    def __init__(self, image_dir, annotation_file, tokenizer_name="EleutherAI/gpt-neo-125M", image_processor_name="openai/clip-vit-base-patch16", max_length=128):
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.image_processor = CLIPImageProcessor.from_pretrained(image_processor_name)
        self.max_length = max_length

        with open(annotation_file, 'r') as f:
            data = json.load(f)

        self.id_to_filename = {img['id']: img['file_name'] for img in data['images']}
        self.samples = [
            {
                "image": self.id_to_filename[ann["image_id"]],
                "caption": ann["caption"]
            }
            for ann in data["annotations"]
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = os.path.join(self.image_dir, sample["image"])
        caption = sample["caption"]

        image = Image.open(image_path).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        encoding = self.tokenizer(caption, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt")
        input_ids = encoding.input_ids.squeeze(0)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels
        }
