import streamlit as st
import pandas as pd
import os
import zipfile
from PIL import Image
from ultralytics import YOLO

from utils import convert_to_xml


CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Iowa State University Colors (adjust as needed)
ISU_CARDINAL = "#C8102E"  # Main Cardinal
ISU_GOLD = "#FFD100"  # Gold Accent
ISU_LIGHT_GRAY = "#EEEEEE"  # Light Gray for backgrounds

PROCESSED_LOG_FILE = "processed_images.csv"


# 1. Load your YOLO model
@st.cache_resource
def load_model():
    '''
        Load the trained YOLO model for object detection.
    Returns:
        ultralytics.YOLO: the YOLO model for object detection.
    '''
    yolo_model = YOLO("021025_yolo11n_best.pt", task="detect")
    yolo_model.eval()
    return yolo_model


model = load_model()


# Streamlit UI
st.set_page_config(
    page_title="CyCounter", page_icon=":microscope:", layout="wide"
)  # Set title and icon

# Header and Title with Styling
st.title("CyCounter")
st.markdown(
    f""" Welcome to **CyCounter**! This app helps you count cells from microscopy images and gives you `XML` files you can edit in ImageJ. """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <style>
        body {{
            background-color: {ISU_LIGHT_GRAY};
        }}
        .stButton>button {{
            color: white;
            background-color: {ISU_CARDINAL}; /* Cardinal button color */
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 5px; /* Rounded corners */
        }}
       .stSlider label {{
            color: {ISU_CARDINAL}; /* Cardinal label color */
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


def load_processed_images():
    '''
    Load the log of processed images from a CSV file.

    Returns:
        pd.DataFrame: a DataFrame containing the log of processed images.
    '''
    if os.path.exists(PROCESSED_LOG_FILE):
        return pd.read_csv(PROCESSED_LOG_FILE)
    return pd.DataFrame(columns=["filename", "date", "full_date"])


def save_processed_images(df):
    '''
    Save the log of processed images to a CSV file.

    Args:
        df (pd.DataFrame): the DataFrame containing the log of processed images.
    '''

    df.to_csv(PROCESSED_LOG_FILE, index=False)


processed_images = load_processed_images()

# 2. Image Upload
uploaded_files = st.file_uploader(
    "Upload Images",
    type=["jpg", "png", "jpeg", "tiff", "tif"],
    accept_multiple_files=True,
)


# 3. Inference and Postprocessing
def process_image(image, conf=0.275, iou=0.5):  
    '''
    Process the uploaded image using the YOLO model and return the center coordinates of the detected objects.
    Args:
        image (BytesIO): the uploaded image file.
        conf (float): the confidence threshold for object detection.
        iou (float): the IOU threshold for object detection.

    Returns:
        list: a list of center coordinates of the detected objects.
    '''
    img = Image.open(image).convert("RGB")
    results = model.predict(img, conf=conf, iou=iou, verbose=False)[
        0
    ]  # Use parameters for confidence and iou

    cell_centers = []
    for box in results.boxes.xyxy:
        x = (box[0] + box[2]) / 2
        y = (box[1] + box[3]) / 2
        # append the center coordinates to the list after converting to integers
        cell_centers.append((int(x), int(y)))
    return cell_centers


# 4. Streamlit UI
process_button = st.button("Process Images")  # Process button

if not uploaded_files:
    st.warning("Please upload an image.")

if uploaded_files and process_button:  # Process only when button is clicked
    xml_files = []
    new_data = []
    for uploaded_file in uploaded_files:
        try:
            centers = process_image(
                uploaded_file, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD
            )  # Pass threshold values

            xml_tree = convert_to_xml(centers, uploaded_file.name)  # Convert to XML
            xml_filename = uploaded_file.name.rsplit(".", 1)[0] + ".xml"  

            xml_files.append({"filename": xml_filename, "tree": xml_tree})

            new_data.append(
                {
                    "filename": uploaded_file.name,
                    "date": pd.Timestamp.date(pd.Timestamp.now()),
                    "full_date": pd.Timestamp.now(),
                }
            )

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
            continue

    if xml_files:
        with zipfile.ZipFile("results.zip", "w") as zipf:
            for xml_file in xml_files:
                xml_file["tree"].write(xml_file["filename"])
                zipf.write(xml_file["filename"])
                os.remove(xml_file["filename"])

        st.download_button(
            label="Download Results (ZIP)",
            data=open("results.zip", "rb"),
            file_name="results.zip",
            mime="application/zip",
        )
        os.remove("results.zip")
        new_df = pd.DataFrame(new_data)
        processed_images = pd.concat([processed_images, new_df], ignore_index=True)
        save_processed_images(processed_images)
        st.success("Processing complete!")
        # st.experimental_rerun() # Rerun to update the plot

# Plotting the data
st.markdown("---")
if not processed_images.empty:
    # Total images processed so far
    total_images = processed_images.shape[0]

    st.markdown(
        f'<div style="text-align: center; font-size: 250px; margin-bottom: -50px;"> <h1>{total_images}</h1> </div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="text-align: center"> <h2>Processed Images</h2> </div>',
        unsafe_allow_html=True,
    )

else:
    st.write("No image processing data yet.")

# Readme Section
st.markdown("---")  # Separator
st.subheader("How to Use CyCounter")
st.write(
    """
    1. **Upload Images:** Click the "Browse files" button to select one or more image files (JPG, PNG, or JPEG format). You can select multiple images by holding down Ctrl (or Cmd on Mac) while clicking.
    2. **Process Images:** After uploading the images, click the "Process Images" button. CyCounter will analyze each image, detect objects, and calculate the center coordinates of the bounding boxes.
    3. **Download Results:** Once the processing is complete, a "Download Results (ZIP)" button will appear. Click it to download a zip file containing individual CSV files for each processed image. Each CSV file will contain the x and y coordinates of the detected object centers.

    **Note:** CyCounter uses fixed confidence and IOU thresholds for object detection.  If you need to adjust these, please contact the developer.

    Made with :hearts: by Abdurahman A. Mohammed 
    """
)
