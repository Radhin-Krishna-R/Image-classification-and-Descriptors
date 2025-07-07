# 🧠 Multi-Model Image Descriptor: A Fusion of CNN & RNN for Visual Understanding

This repository contains the complete implementation of a **self-contained image understanding system** that combines **object classification** and **image captioning**. It is optimized for **offline, low-latency performance** and is suitable for **assistive technologies** like **bionic vision** and **neuro-prosthetics**.

---

## 🔍 Project Description

This project develops a deep learning system that can:
1. **Detect and classify multiple objects** in an image using a CNN-based model.
2. **Generate a natural language caption** describing the image content using a CNN-LSTM model.

Unlike many modern AI systems that rely on APIs or cloud-based models, this solution is designed for **offline** execution on edge devices, ideal for scenarios where **low latency**, **user privacy**, and **network independence** are essential.

---

## 🧠 System Architecture

The system consists of two main modules:
- **Object Detection and Classification**: Based on **YOLOv11n**, trained on the **MS COCO** dataset.
- **Image Captioning**: Implemented using a **CNN-LSTM encoder-decoder model** trained on **Flickr8k**.

Both modules are integrated into a **real-time webcam pipeline**, alongside a **Streamlit web application** for user interaction.

---

## 📂 Components Overview

### 1. Object Classification (YOLOv11n)
- Trained on the COCO dataset (80+ classes)
- Provides bounding boxes and class labels
- Supports both static and webcam-based input
- Achieved ~81% mAP@0.5 with 80% precision and 70% recall

### 2. Image Captioning (CNN + LSTM + Attention)
- **Encoder**: Pretrained InceptionV3 or EfficientNetB4
- **Decoder**: LSTM with Bahdanau attention mechanism
- Trained on Flickr8k for full-sentence outputs
- Achieved ~0.72 BLEU/CIDEr score

### 3. Webcam Integration
- Real-time inference via OpenCV + TensorFlow
- ~50ms latency per frame on mid-tier hardware
- Displays live bounding boxes and labels

### 4. Streamlit Web App
- Upload images/videos or access webcam
- Select detection classes and view predictions
- Offers real-time captions and bounding boxes
- Fully offline and can be hosted locally or via Streamlit Cloud

---

## 🧪 Performance Metrics

| Task              | Metric             | Inference Time | Notes                          |
|-------------------|--------------------|----------------|-------------------------------|
| Object Detection  | 77% Accuracy       | ~120ms/frame   | Evaluated on COCO              |
| Image Captioning  | ~0.72 CIDEr score  | ~180ms/image   | Evaluated on Flickr8k          |
| Webcam Inference  | 30 FPS             | ~50ms/frame    | Using EfficientNetB0 encoder   |
| Web App           | Responsive         | ~100ms latency | Lightweight Streamlit frontend |

---

## 🧰 Technologies & Libraries

| Area              | Tools & Frameworks                        |
|-------------------|-------------------------------------------|
| Deep Learning     | TensorFlow, Keras, Ultralytics YOLO       |
| Image Processing  | OpenCV, Pillow (PIL)                      |
| NLP / Captioning  | Tokenizer, LSTM, Attention Mechanism      |
| Visualization     | Matplotlib, Streamlit                     |
| IDEs              | Jupyter Notebook, Visual Studio Code      |
| Deployment        | Local Hosting / Streamlit Cloud (Optional)|

---

## ⚙️ Installation & Execution

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/image-captioning-cnn-rnn.git
cd image-captioning-cnn-rnn
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Object Classification
```bash
python classify.py
```

### 4. Run Image Captioning
```bash
python caption.py
```

### 5. Launch Streamlit Web App
```bash
streamlit run app.py
```

---

## 📦 Datasets

- **MS COCO**: Used for training object detection (YOLO).
- **Flickr8k**: Used for training the image captioning model.

---

## 🧪 Training Details

**Object Detection (YOLOv11n)**
- Epochs: 30
- Image Size: 640×640
- Final Training Loss: < 0.33

**Image Captioning (CNN-LSTM)**
- Epochs: 50 (with early stopping)
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Batch Size: 64
- Sequence Length: 25

---

## 🔐 Design Philosophy

- **Privacy-First**: Operates without cloud APIs or remote services
- **Offline-Ready**: Fully functional without internet access
- **Real-Time Capable**: Sub-200ms inference pipeline
- **Assistive-Technology Ready**: Designed for accessibility solutions

---

## 🔭 Future Enhancements

- Extend captioning dataset beyond Flickr8k for better generalization
- Upgrade LSTM decoder with Transformer architecture
- Add speech-to-text integration for audio-based interaction
- Optimize models using ONNX or TensorRT for edge deployment

---

## 👤 Author

**Radhin Krishna R**  
B.Sc. Data Science (Reg No: 22376003)  
Hindustan Institute of Technology and Science, Chennai

---


## 🤝 Acknowledgements

- **Dr. I. Lakshmi** – Project Supervisor  
- **Dr. Princy Suganthi Bai** – Head of Department  
- **Dr. B. Nithya** – Project Coordinator

---

## 📚 References

Key inspirations and concepts were adapted from:
- Show and Tell (2015)
- Bahdanau Attention
- YOLOv11n Architecture
- Transformer Decoder Models  
(Full references available in the project report)
