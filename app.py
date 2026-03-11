import streamlit as st
from sqlalchemy import text
import pandas as pd
import os
import zipfile
from PIL import Image
from ultralytics import YOLO
from utils import convert_to_xml

# --- CONFIG & MODEL ---
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
ISU_CARDINAL = "#C8102E"
ISU_GOLD = "#FFD100"
ISU_LIGHT_GRAY = "#EEEEEE"

@st.cache_resource
def load_model():
    yolo_model = YOLO("021625_yolo11m_best.pt", task="detect")
    yolo_model.eval()
    return yolo_model

model = load_model()
conn = st.connection("postgresql", type="sql")

# --- UTILITIES ---
def format_big_number(num):
    """Formats large numbers to be more readable (e.g., 1,234 or 1.2k)."""
    if num >= 1000000:
        return f"{num / 1000000:.1f}M"
    elif num >= 1000:
        return f"{num / 1000:.1f}k"
    else:
        return f"{num:,}"

def log_to_supabase(filename, count):
    try:
        with conn.session as session:
            statement = text("""
                INSERT INTO image_logs (filename, full_date, object_count) 
                VALUES (:name, :date, :count)
            """)
            session.execute(statement, params={
                "name": filename, 
                "date": pd.Timestamp.now(), 
                "count": count
            })
            session.commit()
    except Exception as e:
        st.error(f"Database Logging Error: {e}")

# --- UI SETUP & STYLING ---
st.set_page_config(page_title="CyCounter", page_icon=":microscope:", layout="wide")

# RESTORED ISU COLORS
st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {ISU_LIGHT_GRAY};
        }}
        .stButton>button {{
            color: white !important;
            background-color: {ISU_CARDINAL} !important;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
        }}
        .stButton>button:hover {{
            background-color: {ISU_GOLD} !important;
            color: {ISU_CARDINAL} !important;
        }}
        h1, h2, h3 {{
            color: {ISU_CARDINAL};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("CyCounter")
st.markdown(
    f""" Welcome to **CyCounter**! This app helps you count cells from microscopy images and gives you `XML` files you can edit in ImageJ. """,
    unsafe_allow_html=True,
)

# --- MAIN APP LOGIC ---
uploaded_files = st.file_uploader(
    "Upload Images",
    type=["jpg", "png", "jpeg", "tiff", "tif"],
    accept_multiple_files=True,
)

process_button = st.button("Process Images")

def process_image(image, conf=0.275, iou=0.5):  
    img = Image.open(image).convert("RGB")
    results = model.predict(img, conf=conf, iou=iou, max_det=3000, verbose=False)[0]
    cell_centers = []
    for box in results.boxes.xyxy:
        x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        cell_centers.append((int(x), int(y)))
    return cell_centers

if uploaded_files and process_button:
    xml_files = []
    for uploaded_file in uploaded_files:
        try:
            centers = process_image(uploaded_file, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
            obj_count = len(centers)
            
            xml_tree = convert_to_xml(centers, uploaded_file.name)
            xml_filename = uploaded_file.name.rsplit(".", 1)[0] + ".xml"
            xml_files.append({"filename": xml_filename, "tree": xml_tree})
            
            log_to_supabase(uploaded_file.name, obj_count)
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
            continue

    if xml_files:
        with zipfile.ZipFile("results.zip", "w") as zipf:
            for xml_file in xml_files:
                xml_file["tree"].write(xml_file["filename"])
                zipf.write(xml_file["filename"])
                os.remove(xml_file["filename"])
        
        st.download_button("Download Results (ZIP)", data=open("results.zip", "rb"), file_name="results.zip")
        os.remove("results.zip")
        st.success("Processing complete and logged to database!")

# --- DISPLAY STATS ---
st.markdown("---")
try:
    processed_data = conn.query("SELECT object_count FROM image_logs;", ttl=0)
    
    if not processed_data.empty:
        total_images = len(processed_data)
        total_cells = processed_data["object_count"].sum()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<h1 style='text-align: center; margin-bottom:0;'>{total_images:,}</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: gray;'>Total Images Processed</p>", unsafe_allow_html=True)
        with col2:
            # Using the formatting function for big cell counts
            st.markdown(f"<h1 style='text-align: center; margin-bottom:0;'>{format_big_number(total_cells)}</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: gray;'>Total Cells Detected</p>", unsafe_allow_html=True)
    else:
        st.info("No image processing data found in database yet.")
except Exception as e:
    st.warning("Could not refresh dashboard stats. Database may be sleeping.")

# Readme Section
st.markdown("---")  # Separator
st.subheader("How to Use CyCounter")
st.write(
    """
    1. **Upload Images:** Click the "Browse files" button to select one or more image files (JPG, PNG, or JPEG format). You can select multiple images by holding down Ctrl (or Cmd on Mac) while clicking.
    2. **Process Images:** After uploading the images, click the "Process Images" button. CyCounter will analyze each image, detect objects, and calculate the center coordinates of the bounding boxes.
    3. **Download Results:** Once the processing is complete, a "Download Results (ZIP)" button will appear. Click it to download a zip file containing individual XML files for each processed image. You can load the XML files in ImageJ cell counter to view and edit the counted cells.

    **Note:** CyCounter uses fixed confidence and IOU thresholds for object detection.  If you need to adjust these, please contact the developer.

    Made with :coffee: by Abdurahman A. Mohammed 
    """
)
