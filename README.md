# AutoPlate-Vision-Smart-License-Plate-Detection-and-Recognition-System

## Project Overview
This Streamlit application leverages the YOLO model for detecting objects in images and videos. It supports various file formats for input and generates annotated outputs, making it easy to visualize detected objects with bounding boxes and confidence scores.

## Features
- **Supported Input Formats**:
  - Images: `.jpg`, `.jpeg`, `.png`
  - Videos: `.mp4`, `.mkv`
- **Output**:
  - Annotated images saved locally.
  - Annotated videos saved as `.avi` files.
- User-friendly interface for uploading and processing files.

## Prerequisites
- Python 3.8 or higher
- Required libraries:
  - `streamlit`
  - `numpy`
  - `opencv-python`
  - `Pillow`
  - `ultralytics`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   ```
2. Navigate to the project directory:
   ```bash
   cd your-repo-name
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Upload an image or video file using the provided interface.
3. View or download the processed output:
   - Annotated images can be previewed directly in the app.
   - Annotated videos will be available for download.

## File Structure
- `app.py`: Main Streamlit application.
- `temp/`: Directory for storing temporary uploaded files.
- `output/`: Directory for storing processed files.

## Notes
- Ensure the YOLO model file (`best.pt`) is in the root directory or update the path in the code.
- Temporary and output directories are created automatically during runtime.

## Future Enhancements
- Add support for more file formats.
- Improve real-time video processing capabilities.
- Enhance model performance for larger datasets.

---

Feel free to contribute or provide feedback to improve this project!

