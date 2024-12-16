from datasets import load_dataset
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, ViTImageProcessor

from model import CLIP, CifarDataset


## Parameters 
lr = 1e-5
epochs = 30
batch_size = 64

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print("MPS device not found")

## Data Loading

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

train_dataset = load_dataset("clip-benchmark/wds_vtab-cifar10", split="train")
test_dataset  = load_dataset("clip-benchmark/wds_vtab-cifar10", split="test")

train = CifarDataset(list(train_dataset), processor, tokenizer)
test = CifarDataset(list(test_dataset), processor, tokenizer)

train_loader = DataLoader(train, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test, shuffle=False, batch_size=batch_size)

## Train
device = torch.device("mps" if torch.mps.is_available() else "cpu")
print("Using device: ", device)

model = CLIP().to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
best_loss = np.inf

for epoch in range(epochs):
    loss_per_epoch = 0.0
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
        for i, data in enumerate(train_loader, 0):
            img, cap, mask = data["image"].to(device), data["caption"].to(device), data["mask"].to(device)
            loss = model(img, cap, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Batch Loss": f"{loss.item():.3f}"})
            pbar.update(1)
            
            loss_per_epoch += loss.item()
    loss_per_epoch /= len(train_loader)

    # wandb.log({"epoch": epoch, "loss": loss_per_epoch})

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss_per_epoch:.3f}")

    if loss_per_epoch <= best_loss:
        best_loss = loss_per_epoch
        torch.save(model.state_dict(), "./clip.pt")
        print(f"Model Saved. Loss: {best_loss}")



