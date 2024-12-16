import torch
from PIL import Image
from torch.nn.functional import cosine_similarity
from transformers import DistilBertTokenizer, ViTImageProcessor

from model import CLIP
from test import label_embs, img_embs, get_cls 

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

device = "mps"

# Instantiate the model
model = CLIP()

# Load the model
model.load_state_dict(torch.load("clip.pt", map_location=torch.device("mps")))

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"{device} being used")
model.to(device)

model.eval()
print(type(model).__name__," loaded")

# Text labels for classification
labels = ["dog", "human", "sea otter", "train"]
prompt = "an image of a "

# Tokenize labels and get their embeddings
embedding_dim = 512
num_labels = len(labels)
label_embeddings = torch.zeros(num_labels, embedding_dim).to(device)  # Initialize tensor of zeros

# Compute label embeddings
label_embeddings = label_embs(labels, prompt, model.text_encoder, tokenizer, 512)

# Image inference
img = Image.open("./test.jpg")  # Load image
img_embedding = img_embs(img, model.image_encoder, processor)

# Cosine similarity between the image embedding and the label embeddings
cls_idx = get_cls(img_embedding, label_embeddings, labels)

predicted_class = labels[cls_idx]

print(f"Predicted class: {predicted_class}")
