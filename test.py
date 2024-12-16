import torch
from datasets import load_dataset
from transformers import DistilBertTokenizer, ViTImageProcessor
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import argparse

from model import CLIP, CifarDataset

# Text label embeddings
def label_embs(labels, prompt, encoder, tokenizer, embedding_dim):
    label_embeddings = torch.zeros(len(labels), embedding_dim).to(device)

    for idx, label in enumerate(labels):
        if prompt:
            label = "".join([prompt, label])

        text_e = tokenizer(label, return_tensors="pt", padding=True, truncation=True, max_length=128)
        label_embedding = encoder(text_e['input_ids'].to(device), text_e['attention_mask'].to(device))
        label_embeddings[idx] = label_embedding.detach() 

    return label_embeddings

# Image embeddings
def img_embs(img, encoder, processor):
    img = processor(img, return_tensors="pt")["pixel_values"].to(device)
    img_embedding = encoder(img)

    return img_embedding

# Classification inference
def get_cls_idx(img_embedding, label_embeddings, labels):
    img_embedding_norm = F.normalize(img_embedding, p=2, dim=-1)  
    label_embeddings_norm = F.normalize(label_embeddings, p=2, dim=-1)  

    img_embedding_matched = img_embedding_norm.unsqueeze(1).expand(-1, 10, -1) # dimension 1 expanded by the number of classes
    cos_sim = F.cosine_similarity(img_embedding_matched, label_embeddings_norm, dim=-1)
    
    predicted_class_index = torch.argmax(cos_sim, dim=1)

    return predicted_class_index


if __name__ == "__main__":
    cifar10 = load_dataset("clip-benchmark/wds_vtab-cifar10", split="test")

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    # Instantiate the model
    model = CLIP()

    # Load the model
    model.load_state_dict(torch.load("flickr_clip.pt", map_location=torch.device("mps")))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"{device} being used")
    model.to(device)

    model.eval()
    print(type(model).__name__," loaded")

    # Test dataset to dataloader
    batch_size = 128
    test_dataset = CifarDataset(cifar10, processor, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    label_embeddings = label_embs(test_dataset.labels, False, model.text_encoder, tokenizer, 512).to(device)

    cls_output = []
    with torch.no_grad():  
        for x in tqdm(test_loader, desc="Evaluating", leave=False):
            img_embeddings = model.image_encoder(x['image'].to(device))
            true_cls_idxs = x['cls'].to(device)

            predicted_cls_idxs = get_cls_idx(img_embeddings, label_embeddings, test_dataset.labels)

            # Compare predictions with true labels for accuracy
            cls_output.append((predicted_cls_idxs == true_cls_idxs).float())

    cls_output = torch.cat(cls_output)
    accuracy = cls_output.mean().item()

    print(f"Accuracy: {accuracy * 100:.2f}%")