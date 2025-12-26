import streamlit as st
import ollama
from PIL import Image
import io
import base64
import re
import os

# Page configuration setup
st.set_page_config(
    page_title="Gemma-3 OCR",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items = {
    'Get Help': 'https://blog.google/technology/developers/gemma-3/',
    'Report a bug': "https://github.com/DMuriuki/RAG-Applications/issues",
    'About': """
    ### About This App

    This application extracts structured data from images using Optical Character Recognition (OCR) powered by Google's **Gemma 3 Vision** model.

    It leverages the latest multimodal capabilities of Gemma 3 to accurately interpret and process visual content, transforming images into usable data formats.

    **Key Features:**
    - Image-to-text extraction using advanced vision models
    - Clean and interactive Streamlit interface
    - Seamless support for bug reporting and user feedback

    Built with ❤️ using Python, Streamlit, and Google Gemma 3.

    ---
    [View Gemma 3 Overview](https://blog.google/technology/developers/gemma-3/)  
    [Report an Issue](https://github.com/DMuriuki/RAG-Applications/issues)
    """
}

)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(BASE_DIR, "assets", "gemma3.png")

with open(image_path, "rb") as img_file:
    img_data = base64.b64encode(img_file.read()).decode()

st.markdown(f"""
    # <img src="data:image/png;base64,{img_data}" width="100" style="vertical-align: -12px;"> Gemma-3 OCR
""", unsafe_allow_html=True)

st.markdown("---")

# Descriptive text below title
st.markdown("""
<div style='margin-bottom: 1rem; font-size:16px;'>
    This application allows you to upload an image and extract structured text using OCR powered by 
    <strong>Google's Gemma 3 Vision</strong> model via <strong>Ollama</strong>. 100% Local.
</div>

<div style='font-size:16px;'>
    ⚠️ Please allow a few minutes for the model to run due to local GPU constraints. <br><br>
</div>
""", unsafe_allow_html=True)

# Collapsible "Setup Guide" and "How It Works"
st.markdown("""
<details>
<summary><strong>⚙️ Setup Guide</strong></summary>

<br>

To use this app, you must have **Ollama** installed and running locally with the **Gemma 3 Vision** model.

1. 👉 Download Ollama from: [ollama.com/download](https://ollama.com/download)  
2. 🧐 Pull and run the model using the following command:

```bash
ollama run gemma:3-vision
```

3. ✅ Make sure Ollama remains running in the background while you use the app.

If the model isn't running or available, the app will display an error during image processing.

</details>

<details>
<summary><strong>🔍 How It Works</strong></summary>

<br>

1. 📄 Upload an image (JPG, PNG, etc.) using the **Browse files** button.  
2. ✈️ The app sends the image to your **local Ollama server**.  
3. 🤖 The **Gemma 3 Vision** model processes the image and extracts structured text.  
4. 📋 The extracted text is displayed below the image preview for review or copy-paste.

This workflow allows you to extract data from documents like forms, receipts, invoices, labels, and more — all using a **local, private, offline-capable AI model**.

</details>
""", unsafe_allow_html=True)

st.markdown("---")

# Add clear button to top right
col1, col2 = st.columns([5, 2])

with col2:
    st.write("")  
    if st.button("Clear Uploaded Image 🗑️"):
        if 'ocr_result' in st.session_state:
            del st.session_state['ocr_result']
        st.rerun()

st.markdown('<p style="margin-top: -45px; margin-bottom: 0; padding: 0;">Extract structured text from images using Gemma-3 Vision!</p>', unsafe_allow_html=True)
st.markdown("---")

# Move upload controls to sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("<h2 style='font-weight: 800; color: #ff4d4d; text-align: center'>Upload Image File</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image file...", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.markdown("---")
        st.markdown("<h2 style='font-weight: 800; color: #ff4d4d; text-align: center'>Uploaded Image</h2>", unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image")
        st.markdown("---")
        if st.button("Extract Text 🔍", type="primary"):
            with st.spinner("Processing image..."):
                try:
                    response = ollama.chat(
                        model = "llava",
                        messages=[{
                            'role': 'user',
                            'content': """Analyze the text in the provided image. Extract all readable content
                                        and present it in a structured Markdown format that is clear, concise, 
                                        and well-organized. Ensure proper formatting (e.g., headings, lists, or
                                        code blocks) as necessary to represent the content effectively.""",
                            'images': [uploaded_file.getvalue()]
                        }]
                    )
                    raw_text = response.message.content

                    # Remove unwanted leading line 
                    cleaned_text = re.sub(r"(?i)^here's the extracted.*?:\s*", "", raw_text).strip()
                    st.session_state['ocr_result'] = cleaned_text
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")

# Main content area for results
if 'ocr_result' in st.session_state:
    st.markdown("### Here's the extracted content from the image, presented in a structured Markdown format:")
    styled_md = f"""
    <div style="
        background-color: #520f06;
        border-left: 5px solid #ff4d4d;
        padding: 1em;
        border-radius: 6px;
        margin-top: 1em;
    ">
    {st.session_state['ocr_result']}
    </div>
    """
    st.markdown(styled_md, unsafe_allow_html=True)
else:
    st.info("Upload an image and click 'Extract Text' to see the results here.")

# Footer
st.markdown("---")
st.markdown("Powered by the Gemma-3 Vision Model | [Report an Issue](https://github.com/DMuriuki/RAG-Applications/issues)")
st.markdown("© 2025 Dickson Wanjau")
st.markdown("---")
