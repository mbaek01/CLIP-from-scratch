# Building CLIP from scratch in PyTorch

This project implements a custom CLIP (Contrastive Language-Image Pretraining) model from scratch using PyTorch, with components for training, testing, and dataset handling. 

## Overview
This implementation focuses on building a CLIP model with:

* A Text Encoder using the DistilBERT model.
* An Image Encoder using the ViT (Vision Transformer) model.
* Contrastive learning between image and text embeddings.

It supports the CIFAR-10 dataset for classification tasks and enables training and testing using customized datasets.

## Project Structure
The repository contains the following files:

* model.py: Defines the CLIP model, image/text encoders, and dataset classes.
* train.py: Script for training the CLIP model on a custom dataset.
* test.py: Script for testing the CLIP model and evaluating classification accuracy.
* inference.py: Script that takes a test image and predicts its class label based on predefined text labels.

## Dependencies

```python
pip install torch transformers datasets tqdm numpy
```

## How to Use
### Training

Note: This implementation uses Apple's MPS for GPU acceleration. If you're using a different platform or GPU, make sure to adjust the device settings accordingly (e.g., replace "mps" with "cuda").

Ensure you have the CIFAR-10 dataset or download it via Hugging Face's datasets library.

Run the train.py script:
```bash
python train.py
```

Training parameters such as learning rate, batch size, and number of epochs may be adjusted accordingly. 

### Example of a Loss Curve
<img width="445" alt="Screenshot 2024-12-18 at 12 46 42 AM" src="https://github.com/user-attachments/assets/558d2bdf-7dfd-4478-a6e9-c848b78415df" />

The model checkpoints are saved as clip.pt.

### Testing

Ensure the clip.pt checkpoint is available (created during training).

Run the test.py script:
```bash
python test.py
```
The script computes accuracy on the test dataset and outputs it to the console.

(A small batch size may result in lower accuracy.) 


## Model Components
### CLIP Architecture
The model consists of:

* Text Encoder: A frozen DistilBERT model that converts input text into a 512-dimensional embedding.
* Image Encoder: A ViT model that processes input images into a 512-dimensional embedding.
* Contrastive Loss: Computes a cross-entropy loss between image-text embeddings to learn multimodal representations.
  
### Datasets
* CIFAR-10
* Flickr8k
  
  (Both downloaded from /clip-benchmark from HuggingFace)
