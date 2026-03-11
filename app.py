import streamlit as st
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

# --- SUPABASE CONNECTION ---
# Ensure you have [connections.postgresql] url in your secrets.toml
conn = st.connection("postgresql", type="sql")

def log_to_supabase(filename, count):
    """Logs processing event to Supabase instead of CSV."""
    try:
        with conn.session as session:
            session.execute(
                "INSERT INTO image_logs (filename, full_date, object_count) VALUES (:name, :date, :count);",
                params={
                    "name": filename, 
                    "date": pd.Timestamp.now(), 
                    "count": count
                }
            )
            session.commit()
    except Exception as e:
        st.error(f"Database Logging Error: {e}")

# --- UI SETUP ---
st.set_page_config(page_title="CyCounter", page_icon=":microscope:", layout="wide")
st.title("CyCounter")
st.markdown("Welcome to **CyCounter**! Count cells and export `XML` for ImageJ.")

# (Your existing CSS markdown here...)

# --- IMAGE PROCESSING ---
def process_image(image, conf=0.275, iou=0.5):  
    img = Image.open(image).convert("RGB")
    results = model.predict(img, conf=conf, iou=iou, max_det=3000, verbose=False)[0]

    cell_centers = []
    for box in results.boxes.xyxy:
        x = (box[0] + box[2]) / 2
        y = (box[1] + box[3]) / 2
        cell_centers.append((int(x), int(y)))
    return cell_centers

# --- MAIN APP LOGIC ---
uploaded_files = st.file_uploader(
    "Upload Images",
    type=["jpg", "png", "jpeg", "tiff", "tif"],
    accept_multiple_files=True,
)

process_button = st.button("Process Images")

if uploaded_files and process_button:
    xml_files = []
    
    for uploaded_file in uploaded_files:
        try:
            centers = process_image(uploaded_file, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
            obj_count = len(centers) # Get the count here

            # Generate XML
            xml_tree = convert_to_xml(centers, uploaded_file.name)
            xml_filename = uploaded_file.name.rsplit(".", 1)[0] + ".xml"
            xml_files.append({"filename": xml_filename, "tree": xml_tree})

            # LOG TO SUPABASE
            log_to_supabase(uploaded_file.name, obj_count)

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
            continue

    if xml_files:
        # Zip creation logic
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
        st.success("Processing complete and logged to database!")

# --- DISPLAY STATS FROM SUPABASE ---
st.markdown("---")
# Fetch counts from Supabase to keep the dashboard persistent
try:
    processed_data = conn.query("SELECT * FROM image_logs;", ttl=0)
    
    if not processed_data.empty:
        total_images = len(processed_data)
        total_cells = processed_data["object_count"].sum()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<h1 style='text-align: center;'>{total_images}</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Total Images Processed</p>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<h1 style='text-align: center;'>{total_cells}</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Total Cells Detected</p>", unsafe_allow_html=True)
    else:
        st.write("No image processing data yet.")
except Exception:
    st.write("Waiting for database connection...")
