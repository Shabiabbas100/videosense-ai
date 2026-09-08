import streamlit as st
import os
from utils import faiss_db_utils as faiss
from utils import llm_utils as llm

# """
# This is the main file for the website (frontend).
# It shows a search bar, takes the user's text, finds the video using FAISS, 
# shows the video on the screen, and prints the Gemini AI report.
# """

# Set up the look of the web page
st.set_page_config(page_title="VideoSense AI", page_icon="🎥")

st.markdown(
    "<h1 style='text-align: center;'>🎥 VideoSense AI</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; font-size: 18px; font-style: italic;'>AI-Powered Video Search & Analysis</p>",
    unsafe_allow_html=True
)

# Text box for the user to type what they want to find
query = st.text_input("Enter your search query (example: 'person walking'):")

if st.button("Search"):
    if not query.strip():
        st.warning("Please type a search query first.")
    else:
        with st.spinner("Finding the best video match..."):
            try:
                # 1. Ask FAISS database to find the closest video clip
                retrieved_clips = faiss.search_videos_by_text(query) 
            except Exception as e:
                st.error(f"Error finding the video: {e}")
                retrieved_clips = []
            
            if retrieved_clips:
                st.subheader("🔍 Found Video Clip")
                
                # We show the best matching clip
                best_clip = retrieved_clips[0]
                video_path = f"utils/clips/{best_clip}"
                
                # 2. Show the video on the screen if it exists
                if os.path.exists(video_path):
                    st.write(f"**{best_clip}**")
                    st.video(video_path) 
                else:
                    st.warning(f"❌ Video file is missing from folder: {video_path}")
                
                # 3. Send the video to Gemini AI to get a smart explanation
                st.subheader("🧠 Gemini AI Security Report")
                with st.spinner("Gemini AI is watching the video and writing a report. This takes a few seconds..."):
                    llm_response = llm.validate_with_llm(query, retrieved_clips) 
                    st.write(llm_response) 
            else:
                st.warning("No relevant videos found. Make sure you ran the pipeline to generate clips.")