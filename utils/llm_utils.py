import google.generativeai as genai
import time
import os
from dotenv import load_dotenv

load_dotenv()

def validate_with_llm(query, retrieved_clips):
    """
    Sends the found video clip to Google's Gemini AI so it can watch the video 
    and explain what is happening inside it based on the user's search.
    
    Args:
        query (str): What the user searched for (e.g., "person with a backpack").
        retrieved_clips (list): The list of video names our search engine found.
        
    Returns:
        str: A text report from Gemini explaining the video, or an error message.
    """
    
    # 1. Setup the AI with your secret key
    my_api_key = os.getenv("GOOGLE_API_KEY")
    if not my_api_key:
        return "Error: Google API key is missing. Please check your .env file."
        
    genai.configure(api_key=my_api_key)
    
    # Using Gemini 1.5 Flash because it is very fast and great at watching videos
    llm = genai.GenerativeModel('gemini-2.5-flash')
    
    # Get the best matching video clip (the first one in the list)
    if not retrieved_clips:
        return "No clips were found to analyze."
        
    video_file_name = f"utils/clips/{retrieved_clips[0]}"
    
    if not os.path.exists(video_file_name):
        return f"Error: Video file {video_file_name} not found on the computer."
    
    # 2. Upload the video to Google servers
    print("Uploading video to Gemini AI...")
    uploaded_video = genai.upload_file(path=video_file_name)
    
    # 3. Wait for Google to process the video
    # Videos take a few seconds to load on their servers before we can ask questions
    while uploaded_video.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(2)
        uploaded_video = genai.get_file(uploaded_video.name)
        
    if uploaded_video.state.name == "FAILED":
        return "Error: Gemini AI failed to process the video."

    # 4. Ask the AI to act as a security guard and write a report
    prompt = f"""
    You are an AI security analyst. The user searched for: "{query}". 
    Watch this video and give a clear report in this format:
    - **Summary of Events**: What is happening in the video?
    - **Behavioral Patterns**: How are the people or objects moving?
    - **Potential Threat Assessment**: Is there anything dangerous or suspicious?
    """
    
    try:
        # Send the video and our prompt to Gemini
        contents = [uploaded_video, prompt]
        response = llm.generate_content(contents)
        result = response.text
    except Exception as e:
        result = f"Error during AI analysis: {e}"
        
    # 5. Delete the video from Google's servers to save space and keep data private
    genai.delete_file(uploaded_video.name)
    
    return result