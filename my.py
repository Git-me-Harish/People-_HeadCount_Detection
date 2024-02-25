import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO("yolov8n.pt")

# Define the main function for the Streamlit app
def main():
    # Set title of the Streamlit app
    st.title("Object Detection with YOLO")

    # Sidebar option to select process type
    process_type = st.sidebar.selectbox("Select process type", ["Image", "Video"])

    if process_type == "Image":
        detect_objects_in_image()
    elif process_type == "Video":
        detect_objects_in_video()

def detect_objects_in_image():
    # Upload image file
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Perform object detection if an image is uploaded
    if uploaded_image is not None:
        try:
            # Convert uploaded image to numpy array
            image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(image, -1)

            # Perform object detection
            results = model(image)

            # Annotate image with detected objects
            annotated_image = annotate_image(image, results)

            # Display annotated image
            st.image(annotated_image, caption="Annotated Image", use_column_width=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")

def detect_objects_in_video():
    # Upload video file
    uploaded_video = st.file_uploader("Upload a video", type=["mp4"])

    # Perform object detection if a video is uploaded
    if uploaded_video is not None:
        try:
            # Save the uploaded video to a temporary location
            temp_video_path = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
            with open(temp_video_path, "wb") as temp_video:
                temp_video.write(uploaded_video.read())

            # Perform object detection on the video
            annotated_video_path = detect_objects_on_video(temp_video_path)

            # Display the annotated video
            st.video(annotated_video_path)

        except Exception as e:
            st.error(f"Error: {str(e)}")

def annotate_image(image, results):
    # Draw bounding boxes on the image
    for box in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = box.tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{model.names[int(cls)]}: {conf:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return image

def detect_objects_on_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output_path = os.path.join(tempfile.gettempdir(), "temp_annotated_video.mp4")
    out = cv2.VideoWriter(temp_output_path, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection on the frame
        results = model(frame)

        # Annotate frame with detected objects
        annotated_frame = annotate_image(frame, results)

        # Write annotated frame to video
        out.write(annotated_frame)

    cap.release()
    out.release()

    return temp_output_path

# Run the Streamlit app
if __name__ == "__main__":
    main()
