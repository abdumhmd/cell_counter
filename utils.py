import os
import xml.etree.ElementTree as ET

def convert_to_xml(centers, image_file):
    '''
    Convert the centers of the detected cells to an xml file that is compatible with the ImageJ Cell Counter plugin.
    Args:
        centers (list): list of tuples containing the x and y coordinates of the detected cells.
        image_file (str): the path to the image file.

    Returns:
        xml.etree.ElementTree.ElementTree: the xml tree containing the cell centers.
    '''
    tree = ET.Element("CellCounter_Marker_File")

    image_properties = ET.SubElement(tree, "Image_Properties")
    ET.SubElement(image_properties, "Image_Filename").text = os.path.basename(image_file)
    ET.SubElement(image_properties, "X_Calibration").text = "1.0"
    ET.SubElement(image_properties, "Z_Calibration").text = "1.0"
    ET.SubElement(image_properties, "Calibration_Unit").text = "pixel"
    ET.SubElement(image_properties, "Y_Calibration").text = "1.0"

    marker_data = ET.SubElement(tree, "Marker_Data")
    ET.SubElement(marker_data, "Current_Type").text = "0"
    marker_type = ET.SubElement(marker_data, "Marker_Type")
    ET.SubElement(marker_type, "Type").text = "1"
    ET.SubElement(marker_type, "Name").text = "Type 1"

    for center in centers:
        marker = ET.SubElement(marker_type, "Marker")
        ET.SubElement(marker, "MarkerX").text = str(center[0])
        ET.SubElement(marker, "MarkerY").text = str(center[1])
        ET.SubElement(marker, "MarkerZ").text = str(1)

    tree = ET.ElementTree(tree)

    return tree