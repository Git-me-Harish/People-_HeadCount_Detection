import streamlit as st
import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
import tempfile
import os

# Initialize YOLO model
model = YOLO("yolov8n.pt")

# Define the main function for the Streamlit app
def main():
    # Set title of the Streamlit app
    st.title("Object Detection with YOLO")

    # Sidebar option to select process type
    process_type = st.sidebar.selectbox("Select process type", ["Image", "Video"])

    if process_type == "Image":
        detect_persons_in_image()
    elif process_type == "Video":
        detect_persons_in_video()

def detect_persons_in_image():
    # Upload image file
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Perform object detection if an image is uploaded
    if uploaded_image is not None:
        try:
            # Convert uploaded image to numpy array
            image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)

            # Perform object detection
            results = model(image)[0]
            detections = sv.Detections.from_ultralytics(results)

            # Filter out detections for selected classes (0 and 1)
            selected_classes = [0, 1]
            detections = detections[np.isin(detections.class_id, selected_classes)]

            # Calculate person count
            person_count = sum(1 for class_id in detections.class_id if class_id == 0)  # Assuming class_id 0 corresponds to "person"

            # Initialize annotators
            bounding_box_annotator = sv.BoundingBoxAnnotator()
            label_annotator = sv.LabelAnnotator()

            # Annotate image
            annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

            # Add text indicating the person count
            annotated_image = cv2.putText(annotated_image, f"Person Count: {person_count}", (10, 30),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Display annotated image
            st.image(annotated_image, channels="BGR", caption="Annotated Image", use_column_width=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")

def detect_persons_in_video():
    # Upload video file
    uploaded_video = st.file_uploader("Upload a video", type=["mp4"])

    # Perform object detection if a video is uploaded
    if uploaded_video is not None:
        try:
            # Save the uploaded video to a temporary location
            temp_video_path = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
            with open(temp_video_path, "wb") as temp_video:
                temp_video.write(uploaded_video.read())

            # Process video and save the annotated video
            annotated_video_path = os.path.join(tempfile.gettempdir(), "annotated_video.mp4")
            process_video(temp_video_path, annotated_video_path)

            # Display the annotated video
            st.video(annotated_video_path)

            # Provide download button for the annotated video
            if st.button("Download Annotated Video"):
                with open(annotated_video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                st.download_button(label="Download", data=video_bytes, file_name="annotated_video.mp4")

        except Exception as e:
            st.error(f"Error: {str(e)}")

def process_video(source_path, target_path):
    cap = cv2.VideoCapture(source_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(target_path, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

    # Initialize YOLO objects
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Filter out detections for selected classes (0 and 1)
        selected_classes = [0, 1]
        detections = detections[np.isin(detections.class_id, selected_classes)]

        # Calculate person count
        person_count = sum(1 for class_id in detections.class_id if class_id == 0)  # Assuming class_id 0 corresponds to "person"

        # Annotate frame
        annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        # Add text indicating the person count
        annotated_frame = cv2.putText(annotated_frame, f"Person Count: {person_count}", (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Write annotated frame to video
        out.write(annotated_frame)

    cap.release()
    out.release()

# Run the Streamlit app
if __name__ == "__main__":
    main()