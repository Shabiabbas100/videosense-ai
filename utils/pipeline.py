import os
import cv2
import json
import torch
import subprocess
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

class VideoPipeline:

    """
    This class handles the main video processing work. 
    It cuts a big video into small chunks, finds objects in them using YOLO, 
    and creates search data (embeddings) using CLIP.
    """

    def __init__(self):

        """
        Set up the AI models. 
        We load YOLOv8 to easily find objects and CLIP to understand the visual scenes.
        """

        self.yolo = YOLO('yolov8n.pt')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def run(self, video_path, out_dir="clips", chunk_sec=10):

        """
        The main step that starts the process. 
        First, it quickly cuts the video into 10-second clips. 
        Then, it checks each clip one by one to save memory.
        
        Args:
            video_path (str): The location of your main video file.
            out_dir (str): The folder where the small clips will be saved.
            chunk_sec (int): The length of each small clip in seconds.
        """

        os.makedirs(out_dir, exist_ok=True)
        
        # Fast cut using FFmpeg 'copy'. This does not reduce video quality 
        # and makes sure the video plays perfectly in VS Code.
        print(f"Cutting video into {chunk_sec}-second clips...")
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, 
            "-c", "copy", "-f", "segment", "-segment_time", str(chunk_sec), 
            "-reset_timestamps", "1", f"{out_dir}/clip_%d.mp4"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        clips = sorted([f for f in os.listdir(out_dir) if f.endswith('.mp4')])
        
        for clip_name in clips:
            clip_path = os.path.join(out_dir, clip_name)
            print(f"Processing models on: {clip_name}")
            self._process_clip(clip_path)

    def _process_clip(self, clip_path):
        """
        Reads a 10-second video clip frame by frame. 
        It uses YOLO to name objects and CLIP to get visual numbers (vectors).
        Finally, it mixes them and saves a JSON file next to the video.
        
        Args:
            clip_path (str): The location of the 10-second video clip.
        """
        cap = cv2.VideoCapture(clip_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval = int(fps) # We only process 1 frame every second to save time
        
        v_embeds = []
        s_embeds = []
        objects = set()
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Run YOLO and CLIP once every second
            if frame_count % interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 1. YOLO finds objects (like 'person', 'car')
                res = self.yolo(rgb, verbose=False)
                objs = [self.yolo.names[int(c)] for c in res[0].boxes.cls]
                
                objects.update(objs)
                
                # 2. CLIP turns the image and the object text into vectors
                img_in = self.clip_processor(images=rgb, return_tensors="pt").to(self.device)
                txt_in = self.clip_processor(text=[f"Objects: {', '.join(objs)}"], return_tensors="pt", padding=True).to(self.device)
                
                with torch.no_grad():
                    img_out = self.clip_model.get_image_features(**img_in)
                    txt_out = self.clip_model.get_text_features(**txt_in)
                    
                    # Extract the raw tensor from the Hugging Face wrapper object
                    img_tensor = img_out.pooler_output if hasattr(img_out, 'pooler_output') else (img_out[0] if not isinstance(img_out, torch.Tensor) else img_out)
                    txt_tensor = txt_out.pooler_output if hasattr(txt_out, 'pooler_output') else (txt_out[0] if not isinstance(txt_out, torch.Tensor) else txt_out)
                    
                    v_embeds.append(img_tensor)
                    s_embeds.append(txt_tensor)
            
            frame_count += 1
        cap.release()
        
        if not v_embeds: return # Skip if the video was completely empty
        
        # Average the data over the 10 seconds and format it for search
        print(v_embeds,"this is embedding")
        avg_v = torch.mean(torch.stack(v_embeds), dim=0)
        avg_v /= avg_v.norm(p=2, dim=-1, keepdim=True)
        
        avg_s = torch.mean(torch.stack(s_embeds), dim=0)
        avg_s /= avg_s.norm(p=2, dim=-1, keepdim=True)
        
        # Mix the visual and object data (60% visual importance, 40% object importance)
        fused = (0.6 * avg_v) + (0.4 * avg_s)
        fused /= fused.norm(p=2, dim=-1, keepdim=True)
        
        # Save the result
        json_path = clip_path.replace('.mp4', '_embedding.json')
        with open(json_path, 'w') as f:
            json.dump({
                "filename": os.path.basename(clip_path),
                "primary_objects": list(objects),
                "fused_embedding": fused.cpu().numpy().tolist()[0]
            }, f, indent=2)

if __name__ == "__main__":
    pipe = VideoPipeline()
    pipe.run('../SampleVideo/videoplayback1.mp4')