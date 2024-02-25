import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO('yolov8s.pt')

# Streamlit app
st.title("Object Detection with YOLO")

# Sidebar options
choice = st.selectbox("Select", ["Upload image", "Upload video"])
conf = st.number_input("Confidence threshold", 0.2)

if choice == "Upload image":
    # File uploader for image
    image_data = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if image_data is not None:
        img_summit_button = st.button("Predict")

        if img_summit_button:
            # Open the uploaded image
            image = Image.open(image_data)
            # Save the image temporarily
            image.save("input_data_image.png")

            # Read the image using OpenCV
            frame = cv2.imread("input_data_image.png")

            # Perform object detection
            results = model.predict(source=frame, iou=0.7, conf=conf)
            plot_show = results[0].plot()

            # Display the predicted image
            st.image(plot_show, caption="Predicted Image", use_column_width=True)

elif choice == "Upload video":
    # File uploader for video
    video_file = st.file_uploader("Upload Video", type=["mp4"])

    if video_file is not None:
        vid_summit_button = st.button("Predict")

        if vid_summit_button:
            # Read video file
            video_bytes = video_file.read()
            video_np_array = np.frombuffer(video_bytes, np.uint8)
            cap = cv2.VideoCapture(video_np_array)

            # Video writer
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            out = cv2.VideoWriter(temp_file.name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(cap.get(3)), int(cap.get(4))))

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                # Perform object detection
                results = model.predict(source=frame, iou=0.7, conf=conf)

                # Display the annotated frame
                for res in results:
                    annotated_frame = res.render()
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                    out.write(annotated_frame)

            # Release resources
            cap.release()
            out.release()

            # Display annotated video
            st.video(temp_file.name)

            # Provide download button for the annotated video
            if st.button("Download Annotated Video"):
                with open(temp_file.name, "rb") as video_file:
                    video_bytes = video_file.read()
                st.download_button(label="Download", data=video_bytes, file_name="annotated_video.mp4")
