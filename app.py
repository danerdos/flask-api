import datetime
import traceback
from flask import Flask, json, request, jsonify, send_file
from flask_cors import CORS
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from pydicom.uid import generate_uid
from PIL import Image
import io
import base64
import numpy as np
import cv2
import logging
import torch
import torch.nn as nn
import torchvision.models as models
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_windowing
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load the trained PyTorch model (must match training architecture)
class BoneFractureCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b3()  # Use B3, same as training
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1536, 512),  # B3 has 1536 features
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        return self.model(x)
# Initialize model
device = torch.device("cpu")
cnn_model = BoneFractureCNN().to(device)

# Load state dict with key adjustment
state_dict = torch.load('D:/DANER/Project/python/best_bone_fracture_model2.pth', map_location=device)
cnn_model.load_state_dict(state_dict, strict=True)  # Now keys match exactly!

cnn_model.eval()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# DICOM Upload and Metadata Extraction
@app.route('/api/upload', methods=['POST'])
def upload_dicom():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        logger.debug(f"Uploaded file: {file.filename}")

        # Read the DICOM file
        dicom_data = pydicom.dcmread(io.BytesIO(file.read()), force=True)

        # Decompress if necessary
        if dicom_data.file_meta.TransferSyntaxUID.is_compressed:
            logger.debug("Decompressing DICOM file...")
            dicom_data.decompress()

        if not hasattr(dicom_data, "pixel_array"):
            return jsonify({"error": "The uploaded DICOM file does not contain image data"}), 400

        # Extract metadata
        metadata = {
            "patient_name": str(dicom_data.PatientName) if hasattr(dicom_data, "PatientName") else "Unknown",
            "study_description": str(dicom_data.StudyDescription) if hasattr(dicom_data, "StudyDescription") else "No Description",
            "patient_sex": str(dicom_data.PatientSex) if hasattr(dicom_data, "PatientSex") else "Unknown",
            "patient_id": str(dicom_data.PatientID) if hasattr(dicom_data, "PatientID") else "Unknown",
            "accession_number": str(dicom_data.AccessionNumber) if hasattr(dicom_data, "AccessionNumber") else "Unknown",
            "modality": str(dicom_data.Modality) if hasattr(dicom_data, "Modality") else "Unknown",
            "study_date": str(dicom_data.StudyDate) if hasattr(dicom_data, "StudyDate") else "Unknown",
        }

        # Extract and process pixel data
        pixel_array = apply_modality_lut(dicom_data.pixel_array, dicom_data)
        images = []

        # Check if multi-frame (3D)
        if pixel_array.ndim == 3:
            logger.debug(f"Multi-frame DICOM detected with {pixel_array.shape[0]} frames.")
            for i in range(pixel_array.shape[0]):
                frame = pixel_array[i]
                image = Image.fromarray(frame).convert('L')  # Convert to grayscale
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                images.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
        else:
            logger.debug("Single-frame DICOM detected.")
            image = Image.fromarray(pixel_array).convert('L')
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            images.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))

        metadata["images"] = images  # Attach images to response

        return jsonify(metadata), 200

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500
#prediction
@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        logger.debug(f"Uploaded file for prediction: {file.filename}")

        # Read the DICOM file
        dicom_data = pydicom.dcmread(io.BytesIO(file.read()), force=True)

        # Decompress if necessary
        if dicom_data.file_meta.TransferSyntaxUID.is_compressed:
            logger.debug("Decompressing DICOM file for prediction...")
            dicom_data.decompress()

        if not hasattr(dicom_data, "pixel_array"):
            return jsonify({"error": "The uploaded DICOM file does not contain image data"}), 400

        pixel_array = apply_modality_lut(dicom_data.pixel_array, dicom_data)

        predictions = []

        def preprocess_frame(frame):
            """Preprocess frame to match model training parameters"""
            # Convert to RGB and resize
            resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2RGB)
            
            # Normalize using same parameters as training
            normalized_frame = rgb_frame.astype(np.float32) / 255.0
            normalized_frame = (normalized_frame - 0.5) / 0.5  # Normalize to [-1, 1]
            
            # Convert to tensor and add batch dimension
            tensor_frame = torch.from_numpy(normalized_frame).permute(2, 0, 1).unsqueeze(0)
            return tensor_frame

        # Multi-frame DICOM Handling
        if pixel_array.ndim == 3:
            logger.debug(f"Processing {pixel_array.shape[0]} frames for prediction.")
            for i in range(pixel_array.shape[0]):
                processed_frame = preprocess_frame(pixel_array[i])
                with torch.no_grad():
                    outputs = cnn_model(processed_frame)
                    probabilities = torch.softmax(outputs, dim=1)
                    fracture_prob = probabilities[0][1].item()
                    logger.debug(f"Frame {i+1} - Fracture Probability: {fracture_prob}")

                predictions.append({
                    "frame": i+1,
                    "probability": float(fracture_prob),
                    "classification": "Abnormal" if fracture_prob > 0.5 else "Normal"
                })
        else:
            logger.debug("Processing single-frame DICOM for prediction.")
            processed_frame = preprocess_frame(pixel_array)
            with torch.no_grad():
                outputs = cnn_model(processed_frame)
                probabilities = torch.softmax(outputs, dim=1)
                fracture_prob = probabilities[0][1].item()
                logger.debug(f"Single Frame - Fracture Probability: {fracture_prob}")

            predictions.append({
                "frame": 1,
                "probability": float(fracture_prob),
                "classification": "Abnormal" if fracture_prob > 0.5 else "Normal"
            })
            print(predictions)

        return jsonify({"predictions": predictions}), 200

    except Exception as e:
        logger.error(f"Error processing file for prediction: {str(e)}")
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500

# Add this function to generate the PDF
def generate_metadata_pdf(dicom_data, output_buffer):
    """Helper function to generate PDF from DICOM metadata"""
    doc = SimpleDocTemplate(output_buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    def get_attr(dataset, attr, default="Not Available"):
        try:
            value = getattr(dataset, attr, default)
            return str(value) if value else default
        except:
            return default

    # Title
    story.append(Paragraph("DICOM Metadata Report", styles['Title']))
    story.append(Spacer(1, 12))

    # Add all metadata sections
    sections = [
        ("Patient Information", [
            ("PatientName", get_attr(dicom_data, 'PatientName')),
            ("PatientID", get_attr(dicom_data, 'PatientID')),
            ("PatientSex", get_attr(dicom_data, 'PatientSex')),
            ("PatientBirthDate", get_attr(dicom_data, 'PatientBirthDate')),
            ("PatientAge", get_attr(dicom_data, 'PatientAge')),
            ("PatientWeight", get_attr(dicom_data, 'PatientWeight')),
            ("PatientSize", get_attr(dicom_data, 'PatientSize'))
        ]),
        ("Study Information", [
            ("StudyInstanceUID", get_attr(dicom_data, 'StudyInstanceUID')),
            ("StudyID", get_attr(dicom_data, 'StudyID')),
            ("StudyDate", get_attr(dicom_data, 'StudyDate')),
            ("StudyTime", get_attr(dicom_data, 'StudyTime')),
            ("StudyDescription", get_attr(dicom_data, 'StudyDescription')),
            ("AccessionNumber", get_attr(dicom_data, 'AccessionNumber')),
            ("ReferringPhysicianName", get_attr(dicom_data, 'ReferringPhysicianName')),
            ("StudyPriorityID", get_attr(dicom_data, 'StudyPriorityID'))
        ]),
        ("Series Information", [
            ("SeriesInstanceUID", get_attr(dicom_data, 'SeriesInstanceUID')),
            ("SeriesNumber", get_attr(dicom_data, 'SeriesNumber')),
            ("SeriesDate", get_attr(dicom_data, 'SeriesDate')),
            ("SeriesTime", get_attr(dicom_data, 'SeriesTime')),
            ("SeriesDescription", get_attr(dicom_data, 'SeriesDescription')),
            ("Modality", get_attr(dicom_data, 'Modality')),
            ("BodyPartExamined", get_attr(dicom_data, 'BodyPartExamined')),
            ("ProtocolName", get_attr(dicom_data, 'ProtocolName'))
        ]),
        ("Image Information", [
             ("SOPInstanceUID", get_attr(dicom_data, 'SOPInstanceUID')),
            ("InstanceNumber", get_attr(dicom_data, 'InstanceNumber')),
            ("ImageType", get_attr(dicom_data, 'ImageType')),
            ("Rows", get_attr(dicom_data, 'Rows')),
            ("Columns", get_attr(dicom_data, 'Columns')),
            ("PixelSpacing", get_attr(dicom_data, 'PixelSpacing')),
            ("BitsAllocated", get_attr(dicom_data, 'BitsAllocated')),
            ("WindowCenter", get_attr(dicom_data, 'WindowCenter')),
            ("WindowWidth", get_attr(dicom_data, 'WindowWidth'))
        ]),
        ("Acquisition Parameters", [
            ("AcquisitionDate", get_attr(dicom_data, 'AcquisitionDate')),
            ("AcquisitionTime", get_attr(dicom_data, 'AcquisitionTime')),
            ("AcquisitionNumber", get_attr(dicom_data, 'AcquisitionNumber')),
            ("SliceThickness", get_attr(dicom_data, 'SliceThickness')),
            ("KVP", get_attr(dicom_data, 'KVP')),
            ("ExposureTime", get_attr(dicom_data, 'ExposureTime')),
            ("XRayTubeCurrent", get_attr(dicom_data, 'XRayTubeCurrent')),
            ("ContrastBolusAgent", get_attr(dicom_data, 'ContrastBolusAgent'))
        ]),
        ("Equipment Information", [
            ("Manufacturer", get_attr(dicom_data, 'Manufacturer')),
            ("InstitutionName", get_attr(dicom_data, 'InstitutionName')),
            ("StationName", get_attr(dicom_data, 'StationName')),
            ("ManufacturerModelName", get_attr(dicom_data, 'ManufacturerModelName')),
            ("SoftwareVersions", get_attr(dicom_data, 'SoftwareVersions')),
            ("DeviceSerialNumber", get_attr(dicom_data, 'DeviceSerialNumber'))
        ]),
        ("DICOM Header & File Info", [
            ("SOPClassUID", get_attr(dicom_data, 'SOPClassUID')),
            ("TransferSyntaxUID", get_attr(dicom_data.file_meta, 'TransferSyntaxUID', "Not Available")),
            ("ImplementationClassUID", get_attr(dicom_data.file_meta, 'ImplementationClassUID', "Not Available")),
            ("InstanceCreationDate", get_attr(dicom_data, 'InstanceCreationDate')),
            ("InstanceCreationTime", get_attr(dicom_data, 'InstanceCreationTime'))
        ]),
        ("Advanced & Specialized Metadata", [
            ("PatientOrientation", get_attr(dicom_data, 'PatientOrientation')),
            ("Laterality", get_attr(dicom_data, 'Laterality')),
            ("ViewPosition", get_attr(dicom_data, 'ViewPosition')),
            ("AnatomicRegionSequence", get_attr(dicom_data, 'AnatomicRegionSequence')),
            ("RadiationDose", get_attr(dicom_data, 'RadiationDose'))
        ]),
    ]

    for section_title, items in sections:
        story.append(Paragraph(f"<b>{section_title}</b>", styles['Heading2']))
        for item_name, item_value in items:
            story.append(Paragraph(f"<b>{item_name}:</b> {item_value}", styles['BodyText']))
        story.append(Spacer(1, 12))

    doc.build(story)

# Update the convert_dicom endpoint
@app.route('/api/convert-dicom', methods=['POST'])
def convert_dicom():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        logger.debug(f"Conversion request for: {file.filename}")
        
        # Read DICOM with force=True to handle malformed files
        dicom_data = pydicom.dcmread(io.BytesIO(file.read()), force=True)
        
        # Handle decompression
        if hasattr(dicom_data, 'file_meta') and dicom_data.file_meta.TransferSyntaxUID.is_compressed:
            dicom_data.decompress()
        
        # Verify pixel data exists
        if not hasattr(dicom_data, 'pixel_array'):
            return jsonify({"error": "DICOM contains no image data"}), 400

        # Generate metadata PDF
        pdf_buffer = io.BytesIO()
        generate_metadata_pdf(dicom_data, pdf_buffer)
        pdf_buffer.seek(0)

        # Return the PDF as a downloadable file
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name="dicom_metadata_report.pdf",
            mimetype="application/pdf"
        )

    except Exception as e:
        logger.error(f"DICOM conversion failed: {str(e)}", exc_info=True)
        return jsonify({
            "error": f"Conversion failed: {str(e)}",
            "stacktrace": str(traceback.format_exc()) if app.debug else None
        }), 500


@app.route('/api/convert-to-dicom-with-metadata', methods=['POST'])
def convert_to_dicom_with_metadata():
    if 'file' not in request.files or 'metadata' not in request.form:
        return jsonify({"error": "Missing file or metadata"}), 400

    file = request.files['file']
    metadata = json.loads(request.form['metadata'])

    try:
        # Read image file
        img = Image.open(io.BytesIO(file.read()))
        if img.mode != 'L':
            img = img.convert('L')

        # Create basic DICOM file meta information
        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.7'  # Secondary Capture
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'  # Explicit VR Little Endian
        file_meta.ImplementationClassUID = generate_uid()

        # Create DICOM dataset
        ds = pydicom.FileDataset('converted.dcm', {}, file_meta=file_meta, preamble=b"\0"*128)
        
        # Set required DICOM metadata with defaults
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()
        
        # Set image parameters
        ds.Rows = img.height
        ds.Columns = img.width
        ds.SamplesPerPixel = 1
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelData = np.array(img).tobytes()

        # Set study datetime if not provided
        study_date = metadata.get('StudyDate', datetime.datetime.now().strftime('%Y%m%d'))
        study_time = metadata.get('StudyTime', datetime.datetime.now().strftime('%H%M%S'))
        ds.StudyDate = study_date
        ds.StudyTime = study_time
        ds.AccessionNumber = metadata.get('AccessionNumber', '')
        
        # Set patient information
        ds.PatientName = metadata.get('PatientName', 'Anonymous')
        ds.PatientID = metadata.get('PatientID', '000000')
        ds.PatientSex = metadata.get('PatientSex', 'O')
        
        # Set other metadata
        ds.Modality = metadata.get('Modality', 'OT')
        ds.StudyDescription = metadata.get('StudyDescription', 'Converted from image')
        ds.SeriesDescription = metadata.get('SeriesDescription', 'Converted series')

        # Validate and save
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        
        # Save to buffer
        buffer = io.BytesIO()
        ds.save_as(buffer, write_like_original=False)
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name="converted.dcm",
            mimetype='application/dicom'
        )

    except Exception as e:
        logger.error(f"DICOM conversion failed: {str(e)}", exc_info=True)
        return jsonify({"error": f"Conversion failed: {str(e)}"}), 500
if __name__ == '__main__':
    app.run(debug=True)