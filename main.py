import streamlit as st
import os
from utils import faiss_db_utils as faiss
from utils import llm_utils as llm


st.set_page_config(page_title="VideoSense AI", page_icon="🎥")

st.markdown(
    "<h1 style='text-align: center;'>🎥 VideoSense AI</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; font-size: 18px; font-style: italic;'>AI-Powered Video Analytics & Search</p>",
    unsafe_allow_html=True
)

query = st.text_input("Enter your search query:")

if st.button("Search"):
    with st.spinner("Retrieving videos..."):
        retrieved_clips = faiss.search_videos_by_text(query)  # Retrieve videos based on query usig vector search 
        llm_response = llm.validate_with_llm(query, retrieved_clips)  # Validat and rank with LLM (with opinion),any llm can be used like chatgpt or gemini

        if retrieved_clips:
            st.subheader("🔍 Retrieved Video Clips")
            for clip in retrieved_clips:
                video_path = f"utils/clips/{clip}"
                if os.path.exists(video_path):
                    st.write(f"**{clip}**")
                    st.video(video_path)  # Display the video.
                else:
                    st.warning(f"❌ Video file missing: {video_path}")  # it is handling missing files.
        else:
            st.warning("No relevant videos found.")  # this tell user that no video is available.

        st.subheader("🧠 LLM Validation, Ranking, and Opinion")
        st.write(llm_response)  # Display LLM's response.