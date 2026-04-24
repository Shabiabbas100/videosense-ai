# 🎥 VideoSense AI

> **AI-Powered Multimodal RAG Video Analytics & Search Engine**

VideoSense AI is an intelligent video search and surveillance system that helps users find specific moments in footage using **natural language queries** such as *"man trying to enter"* or *"red car moving fast"*.

Instead of manually reviewing hours of video, the system uses a **Multimodal Retrieval-Augmented Generation (RAG)** pipeline to quickly retrieve relevant frames and validate them with an LLM.

## 🚀 Key Features

- **Natural Language Search:** Search for actions, objects, or events in videos using simple everyday language.
- **Multimodal Embeddings:** Combines visual embeddings from **CLIP** with semantic embeddings from **SentenceTransformers** for more accurate retrieval.
- **Object Detection:** Uses **YOLOv3** to detect and catalog objects in video frames.
- **Fast Vector Retrieval:** Powered by **FAISS (Facebook AI Similarity Search)** for fast similarity search across thousands of embeddings.
- **LLM Validation:** Integrates **Google Gemini Vision API** to analyze retrieved clips and generate structured threat/security insights.
- **Clean UI:** Built with **Streamlit** for a simple, Google-Search-like user experience.

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Computer Vision:** OpenCV, YOLOv3
- **AI / Transformer Models:** PyTorch, Hugging Face (`openai/clip-vit-base-patch32`, `all-MiniLM-L6-v2`)
- **Vector Database:** FAISS
- **LLM Integration:** Google Generative AI (Gemini 2.5 Flash)

## 💻 Installation & Setup

Follow these steps to run the project locally:

### 1. Clone the repository
    ```bash
    git clone https://github.com/YourUsername/videosense-ai.git
### 2. Create a Virtual Environment
    ```bash
    python -m venv project_venv
    source project_venv/Scripts/activate  # On Windows Git Bash
### 3. Install Dependencies
    ```bash
    pip install -r requirements.txt
### 4. Set up Environment Variables: Create a .env file and add your API Key
    ```bash
    GOOGLE_API_KEY=your_api_key_here
### 5. Run the Application
    ```bash
    python -m streamlit run main.py
## 👨‍💻 Author

**Shabi Abbas**  
- GitHub: https://github.com/Shabiabbas100  
- LinkedIn: https://www.linkedin.com/in/shabiabbas100
## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!🙏
