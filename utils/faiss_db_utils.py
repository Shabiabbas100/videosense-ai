import os
import json
import faiss
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

# Global variables to store models in memory so they only load once
_clip_model = None
_clip_processor = None
_faiss_index = None
_video_metadata = []

def load_models():
    """
    Loads the CLIP AI model into the computer's memory.
    We only load it once when someone actually searches for a video, 
    so the app starts up very fast and saves memory.
    """
    global _clip_model, _clip_processor
    if _clip_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_model, _clip_processor

def build_faiss_index(folder_path="utils/clips"):
    """
    Reads all the JSON files saved by our video pipeline and puts them into a fast search database (FAISS).
    
    Args:
        folder_path (str): The folder where our video clips and JSON files are saved.
    """
    global _faiss_index, _video_metadata
    _video_metadata = []
    embeddings_list = []
    
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} not found.")
        return

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith("_embedding.json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Grab the mixed vector (visual + objects) we made earlier
            if "fused_embedding" in data:
                embeddings_list.append(data["fused_embedding"])
                _video_metadata.append(data["filename"])
    
    if not embeddings_list:
        print("No video data found to save.")
        return
        
    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    
    # We use Cosine Similarity (IndexFlatIP) to match vectors.
    # This finds the closest meaning between the search text and the video.
    _faiss_index = faiss.IndexFlatIP(512) 
    _faiss_index.add(embeddings_array)
    
    faiss.write_index(_faiss_index, "video_embeddings.index")
    with open("video_filenames.json", "w") as f:
        json.dump(_video_metadata, f)
        
    print(f"Successfully saved {len(embeddings_list)} videos in the search database.")

def get_text_embedding(text):
    """
    Takes the words a user types (like 'a man running') and turns them into a number vector.
    This allows us to match the text with the video vectors.
    
    Args:
        text (str): The search query typed by the user.
        
    Returns:
        numpy array: A 512-number vector representing the text's meaning.
    """
    model, processor = load_models()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        text_out = model.get_text_features(**inputs)
        
        # Extract the raw tensor safely 
        text_tensor = text_out.pooler_output if hasattr(text_out, 'pooler_output') else (text_out[0] if not isinstance(text_out, torch.Tensor) else text_out)
        
        # Normalize the vector so it matches the video math perfectly
        text_tensor /= text_tensor.norm(p=2, dim=-1, keepdim=True)
        
    return text_tensor.cpu().numpy().astype("float32")

def search_videos_by_text(query_text, top_k=1, folder_path="utils/clips"):
    """
    Searches the database for the most similar video clip based on the user's text.
    
    Args:
        query_text (str): What the user is looking for.
        top_k (int): How many video results to return.
        folder_path (str): Where the clips are saved.
        
    Returns:
        list: The names of the best matching MP4 video files.
    """
    global _faiss_index, _video_metadata
    
    # Build or load the database if it is not ready
    if _faiss_index is None:
        if not os.path.exists("video_embeddings.index"):
            build_faiss_index(folder_path)
        else:
            _faiss_index = faiss.read_index("video_embeddings.index")
            with open("video_filenames.json", "r") as f:
                _video_metadata = json.load(f)

    if _faiss_index is None or _faiss_index.ntotal == 0:
        print("Database is empty.")
        return []
        
    # Turn search text into a vector and find the closest match
    query_embedding = get_text_embedding(query_text)
    distances, indices = _faiss_index.search(query_embedding, top_k)
    
    results = []
    for i in indices[0]:
        if i != -1 and i < len(_video_metadata):
            results.append(_video_metadata[i])
            
    return results

if __name__ == "__main__":
    build_faiss_index("clips")
    print("Test Search Result:", search_videos_by_text("person"))