import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import DistilBertModel, ViTModel


class TextEncoder(nn.Module):
    def __init__(self, proj_dim):
        super().__init__()
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # Freeze BERT parameters
        for param in self.model.parameters():
            param.requires_grad = False  

        self.projection = nn.Linear(self.model.config.hidden_size, proj_dim)
        self.layer_norm = nn.LayerNorm(proj_dim)
    
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            x = outputs.last_hidden_state[:, 0, :]

        # Projecting to wanted dimension
        projected_x = self.projection(x) 

        return self.layer_norm(projected_x)

class ImageEncoder(nn.Module):
    def __init__(self, proj_dim):
        super().__init__()
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224") # outputs hidden state

        for param in self.model.parameters():
            param.requires_grad = True

        # proj_dim = 768
        self.projection = nn.Linear(self.model.config.hidden_size, proj_dim)
        self.layer_norm = nn.LayerNorm(proj_dim)

    def forward(self, x):
        with torch.no_grad():
            outputs = self.model(x) 
        
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_dim)
        projected_embedding = self.projection(cls_embedding)  # Shape: (batch_size, proj_dim)

        return self.layer_norm(projected_embedding)

class InfoNCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.temperature = nn.Parameter(torch.ones([])*np.log(1/7)).to(self.device)

    def forward(self, feature_img, feature_txt):
        logits = feature_img@feature_txt.T * torch.exp(self.temperature)

        labels = torch.arange(feature_img.size(0)).to(self.device)

        loss_img = F.cross_entropy(logits.T, labels)
        loss_txt = F.cross_entropy(logits, labels)

        loss = (loss_img + loss_txt)/2.0 

        return loss
        
class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder(proj_dim=512)
        self.text_encoder = TextEncoder(proj_dim=512)
        self.loss_fn = InfoNCELoss()

    def forward(self, img, cap, mask):
        feature_img = self.image_encoder(img)
        feature_txt = self.text_encoder(cap, mask)
        loss = self.loss_fn(feature_img, feature_txt)

        return loss

class FlickrDataset(Dataset):
    def __init__(self, dataset, processor, tokenizer):
        self.dataset = dataset 
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx]

        # Process image
        img = x["jpg"]
        img = self.processor(img, return_tensors="pt")["pixel_values"].squeeze(0)

        # Tokenize text 
        text = x["txt"].split('\n')[0]
        text_e = self.tokenizer(text, add_special_tokens=True, padding="max_length", truncation=True, return_tensors="pt")

        return {
            "image": img,
            "caption": text_e["input_ids"].squeeze(0),
            "mask": text_e["attention_mask"].squeeze(0)
        }

class MSCOCODataset(FlickrDataset):
    pass

class CifarDataset(Dataset):
    def __init__(self, dataset, processor, tokenizer):
        self.dataset = dataset 
        self.processor = processor
        self.tokenizer = tokenizer
        self.labels = ["a photo of an airplane", 
                        "a photo of an automobile", 
                        "a photo of a bird", 
                        "a photo of a cat", 
                        "a photo of a deer", 
                        "a photo of a dog", 
                        "a photo of a frog", 
                        "a photo of a horse", 
                        "a photo of a ship", 
                        "a photo of a truck"]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x = self.dataset[idx]
        img = x['webp']
        cls = x['cls']

        img = self.processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
        txt =  self.tokenizer(self.labels[cls], add_special_tokens=True, padding="max_length", truncation=True, return_tensors="pt")

        return {
            "image": img,
            "caption": txt["input_ids"].squeeze(0),
            "mask": txt["attention_mask"].squeeze(0),
            "cls": cls
        }


