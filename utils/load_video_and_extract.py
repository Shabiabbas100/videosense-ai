import os
import cv2
import subprocess
import shutil

def extract_and_convert_clips(input_video_path, output_folder, clip_duration=10):
    """
    Extracts clips of a specified duration from an input video, converts them to H.264 using FFmpeg, 
    and saves them to an output folder.

    Args:
        input_video_path (str): The path to the input video file.
        output_folder (str): The path to the output folder where clips will be saved.
        clip_duration (int, optional): The duration of each clip in seconds. Defaults to 10.

    Returns:
        list: A list of paths to the extracted and converted clip files.
    """

    # Start with a clean directory to avoid mixing old and new clips
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder) 
    os.makedirs(output_folder, exist_ok=True) 

    video = cv2.VideoCapture(input_video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_clip = int(clip_duration * fps)
    
    output_clips = []
    current_frame = 0
    clip_count = 0
    
    while current_frame < total_frames:
        temp_clip_filename = os.path.join(output_folder, f'temp_clip_{clip_count}.mp4')
        fixed_clip_filename = os.path.join(output_folder, f'clip_{clip_count}.mp4')

        # Use mp4v codec for intermediate raw file creation
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(temp_clip_filename, fourcc, fps, 
                               (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        
        for _ in range(frames_per_clip):
            ret, frame = video.read()
            if not ret: 
                break
            out.write(frame) 
            current_frame += 1 
        
        out.release()
        
        # Convert raw clip to H.264 using FFmpeg for Gemini/Web browser compatibility
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", temp_clip_filename,
            "-vcodec", "libx264", "-acodec", "aac", "-strict", "experimental",
            "-preset", "fast", "-b:v", "1000k", "-b:a", "128k", fixed_clip_filename
        ]
        # Suppress FFmpeg terminal output to keep the console clean
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        os.remove(temp_clip_filename)
        output_clips.append(fixed_clip_filename)
        clip_count += 1

    video.release()
    return output_clips

# Execution
input_video = '../SampleVideo/videoplayback1.mp4' 
clips_folder = 'clips' 
extract_and_convert_clips(input_video, clips_folder)