import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import json
import os
import faiss


"""
Create a FAISS index from a list of video embeddings.

Args:
    embeddings_data (list): A list of dictionaries containing video embeddings.

Returns:
    faiss.IndexFlatL2: A FAISS index for video retrieval.
"""

folder_path="utils/clips"

json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

embeddings_list = []
filenames = []

for file in json_files:
    file_path = os.path.join(folder_path, file)
    with open(file_path, "r") as f:
        data = json.load(f)
    
    if "visual_embedding" in data and isinstance(data["visual_embedding"], list):
        embedding = np.array(data["visual_embedding"], dtype=np.float32)
        if embedding.shape[0] == 512:  # Ensure correct dimension
            embeddings_list.append(embedding)
            filenames.append(data["filename"])

embeddings_array = np.vstack(embeddings_list)  # Shape: (N, 512)

dimension = 512
index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean) search
index.add(embeddings_array)

faiss.write_index(index, "video_embeddings.index")

# Save filenames to match search results later
with open("video_filenames.json", "w") as f:
    json.dump(filenames, f)

print(f"Stored {len(embeddings_list)} video embeddings in FAISS.")

# FAISS Configuration: Ensure single-threading to prevent potential crashes.
faiss.omp_set_num_threads(1)

with open("video_filenames.json", "r") as f:
    filenames = json.load(f)

folder_path = "utils/clips"
embeddings_data = []

for filename in os.listdir(folder_path):
    if filename.endswith("_embedding.json"):  
        with open(os.path.join(folder_path, filename), "r") as f:
            video_data = json.load(f)
            embeddings_data.append(video_data)

print(f"Loaded {len(embeddings_data)} video embeddings.")

embeddings = np.array([d["visual_embedding"] for d in embeddings_data], dtype=np.float32)
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

faiss.write_index(index, "video_embeddings.index")
print("FAISS index saved.")


def get_text_embedding(text):
    """
    Convert a text query into a CLIP embedding.

    Args:
        text (str): The text query to encode.

    Returns:
        numpy.ndarray: The text embedding as a NumPy array.
    """
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("CLIP model loaded.")
    print("Text query:", text)
    
    inputs = processor(text=text, return_tensors="pt", padding=True)

    # Move model and inputs to GPU if available for faster processing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    text_embedding = clip_model.get_text_features(**inputs)

    # Safety Check: Extract the actual tensor from the Box/Object
    if hasattr(text_embedding, 'text_embeds'):
        text_embedding = text_embedding.text_embeds
    elif hasattr(text_embedding, 'pooler_output'):
        text_embedding = text_embedding.pooler_output

    return text_embedding.detach().cpu().numpy().astype("float32")

def search_videos_by_text(query_text, top_k=1):
    """
    Search for videos in the FAISS index using a text query.

    Args:
        query_text (str): The text query for video retrieval.
        top_k (int): The number of top results to retrieve.

    Returns:
        list: A list of filenames corresponding to the top matching videos.
    """
    query_embedding = get_text_embedding(query_text)
    print("Query embedding shape:", query_embedding.shape)
    print("FAISS index dimension:", index.d)
    
    distances, indices = index.search(query_embedding, top_k)
    similar_videos = [embeddings_data[i]["filename"] for i in indices[0] if i < len(embeddings_data)]
    
    print(f"Found {len(similar_videos)} similar videos.")
    print(f"Distances: {distances}")
    
    return similar_videos