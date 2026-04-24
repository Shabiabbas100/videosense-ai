import google.generativeai as genai
import time
import os
from dotenv import load_dotenv

load_dotenv()

def validate_with_llm(query, retrieved_clips):
    """
    Validates a retrieved video clip against a user query using Google's Gemini Vision AI.
    
    This function uploads the most relevant video clip to Google's servers using the File API
    to bypass payload limits. It waits for the backend processing to complete, then asks 
    the Gemini model to act as a security analyst and provide a structured report based 
    on the user's search query. Finally, it deletes the video from the server to maintain hygiene.

    Args:
        query (str): The natural language search query provided by the user (e.g., "man trying to enter").
        retrieved_clips (list): A list of filenames returned by the FAISS database. The function currently processes the top match at index 0.

    Returns:
        str: A structured markdown report from Gemini containing 'Summary of Events', 'Behavioral Patterns', and 'Potential Threat Assessment'.
             Returns an error string if the video processing fails on Google's end.
    """
    
    my_api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=my_api_key)
    
    # Using the latest stable flash model for faster response times and generous free-tier limits.
    llm = genai.GenerativeModel('gemini-2.5-flash')
    video_file_name = f"utils/clips/{retrieved_clips[0]}"
    
    # 1. Upload the video using File API (Instead of Base64) to prevent 429 Quota Exceeded errors.
    print("Uploading video to Gemini servers...")
    uploaded_video = genai.upload_file(path=video_file_name)
    
    # 2. Wait for Gemini to process the video backend. The File API requires the state to be 'ACTIVE' before generation.
    while uploaded_video.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(2)
        uploaded_video = genai.get_file(uploaded_video.name)
        
    if uploaded_video.state.name == "FAILED":
        return "Error: Video processing failed on Google's servers."

    prompt = f"""
    You are an AI security analyst. Given the user's query: "{query}", analyze the video.
    Extract and present the insights in a structured format:
    - **Summary of Events**
    - **Behavioral Patterns**
    - **Potential Threat Assessment**
    """
    
    # 3. Send the file object directly to the model
    contents = [uploaded_video, prompt]
    response = llm.generate_content(contents)
    
    # 4. Clean up (Delete the video from Google's server to save space and ensure privacy)
    genai.delete_file(uploaded_video.name)
    
    return response.text