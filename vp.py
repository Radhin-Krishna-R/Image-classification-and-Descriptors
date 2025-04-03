import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import numpy as np


model = YOLO("yolo11n.pt") 


st.title("Video Object Detection")
st.image("image.png", caption="object that can be detected", use_container_width=True)
st.write("Upload a video for real-time object detection")

   
with st.sidebar:
    st.header("Settings")
    confidence = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    class_names = list(model.names.values()) 
    selected_classes = st.multiselect(
        "Classes to detect (leave empty for all)",
        class_names
    )

    box_thickness = st.slider("Bounding box thickness", 1, 10, 2)
    text_thickness = st.slider("Text thickness", 1, 5, 1)

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        class_ids = [k for k, v in chosen_model.names.items() if v in classes]
        results = chosen_model.predict(img, classes=class_ids, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)
    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf)
    for result in results:
        for box in result.boxes:
            coords = box.xyxy.cpu().numpy().astype(int)[0]  # Convert to int
            cls_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            
            label = f"{chosen_model.names[cls_id]} {conf_score:.2f}"
            cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, label, (coords[0], coords[1] - 10), cv2.FONT_HERSHEY_PLAIN, text_thickness, (255, 0, 0), text_thickness)   
    return img, results

def create_video_writer(video_cap, output_filename):
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

if model:
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Save uploaded file to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()
        
        # Output file
        output_filename = "output.mp4"
        
        # Open video
        cap = cv2.VideoCapture(tfile.name)
        writer = create_video_writer(cap, output_filename)
        
        # Display placeholders
        video_placeholder = st.empty()
        status_text = st.empty()
        
        if st.button("Process Video"):
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_bar = st.progress(0)
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                # Process frame
                result_img, _ = predict_and_detect(
                    model,
                    frame,
                    classes=selected_classes,
                    conf=confidence,
                    rectangle_thickness=box_thickness,
                    text_thickness=text_thickness
                )
                
                # Write to output
                writer.write(result_img)
                
                # Display
                video_placeholder.image(result_img, channels="BGR", use_column_width=True)
                
                # Update progress
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                progress_bar.progress(min(current_frame / frame_count, 1.0))
            
            # Release resources
            cap.release()
            writer.release()
            
            # Offer download
            st.success("Processing complete!")
            with open(output_filename, "rb") as f:
                st.download_button(
                    "Download processed video",
                    f,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
            
            # Clean up
            os.unlink(tfile.name)
            os.unlink(output_filename)
else:
    st.warning("Please load a valid YOLO model first")
