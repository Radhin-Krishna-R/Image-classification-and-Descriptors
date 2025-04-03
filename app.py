import io
import os
import streamlit as st
import requests
from PIL import Image
from model import get_caption_model, generate_caption

"""
# Image Captioning: Challenges and Advancements

Image captioning, the process of automatically generating a textual description of an image, has been a longstanding challenge in artificial intelligence. Before 2015, models relied primarily on rule-based approaches or shallow machine learning techniques, which struggled with understanding complex visual scenes. The introduction of deep learning, particularly Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs), revolutionized the field by enabling end-to-end learning of image representations and sequential text generation.

### Advances in Image Captioning
Recent breakthroughs include:
- **CNN-LSTM Architectures**: Feature extraction via CNNs like ResNet and InceptionV3, followed by sequence generation using LSTMs.
- **Attention Mechanisms**: Enabling models to focus on relevant image regions while generating captions.
- **Transformer Models**: Replacing LSTMs with self-attention-based architectures like Vision Transformers (ViTs) and BERT-inspired models for improved coherence.
- **Pre-trained Vision-Language Models**: Models such as CLIP and BLIP integrate large-scale vision-language training, significantly enhancing caption quality.
"""
st.image("2-Figure1-1.png", caption="Image Captioning Architecture", use_container_width=True)

"""
### Challenges
Despite these advancements, challenges remain:
- **Ambiguity in Captions**: A single image can have multiple correct descriptions.
- **Data Bias**: Models trained on specific datasets may struggle with diverse real-world images.
- **Computational Cost**: Training large-scale models demands significant resources.

This application leverages a CNN for feature extraction and an LSTM for sequence generation, incorporating attention mechanisms for improved accuracy. Below is the implementation of the image captioning model in a Streamlit web app.
"""

@st.cache_resource
def get_model():
    return get_caption_model()

st.title('Image Captioner')
caption_model = get_model()

def predict():
    captions = []
    try:
        pred_caption = generate_caption('tmp.jpg', caption_model)
        st.markdown('#### Predicted Captions:')
        captions.append(pred_caption)

        for _ in range(2):
            pred_caption = generate_caption('tmp.jpg', caption_model, add_noise=True)
            if pred_caption not in captions:
                captions.append(pred_caption)
        
        for c in captions:
            st.write(c)
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")



# File upload
img_upload = st.file_uploader(label='Upload Image', type=['jpg', 'png', 'jpeg'])
if img_upload:
    try:
        img = Image.open(io.BytesIO(img_upload.read()))
        img = img.convert('RGB')
        st.image(img)
        img.save('tmp.jpg')
        predict()
        os.remove('tmp.jpg')
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")