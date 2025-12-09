Intaglio State Comparator
A Computer Vision Tool for Comparative Analysis of Intaglio Print Impressions
Overview
The Intaglio State Comparator is a computer-vision research tool designed for connoisseurship, technical art history, and print-room analysis.
Given two digitized impressions of the same intaglio plate (etching, engraving, drypoint, mezzotint, etc.), the tool:
Standardizes both images
Aligns the second impression to the first
Computes pixel-level difference maps
Generates interpretive overlays highlighting differences in linework, tone, and plate condition
Supports interactive exploration (thresholds, inversion, visibility toggles)
The tool is optimized for:
Comparing different states of a plate
Studying plate wear, rework, foul biting, drypoint burr
Identifying strengthened or effaced lines
Examining differences between impressions from different museums
Supporting cataloging, catalogues raisonnés, and curatorial decision-making
It does not assign states automatically — instead, it surfaces areas that warrant close looking.
Technology Stack
This tool is written in Python and built on high-performance open-source libraries used in scientific imaging.
Core Libraries
1. OpenCV (cv2)
Handles all major computer-vision operations:
Image decoding (JPEG/PNG/TIFF → arrays)
Grayscale conversion
Resizing & normalization
Feature detection and matching (AKAZE/SIFT)
Homography estimation (RANSAC)
Image warping & alignment
Edge extraction (Canny)
Difference-map generation
Masking & overlay creation
Color conversions (BGR/RGB/HSV)
OpenCV enables fast, vectorized operations suitable for large museum-quality scans.
2. NumPy
Provides efficient array-based computation for:
Pixel arithmetic
Overlay tinting (red/cyan)
Mask and threshold operations
Normalization
Channel stacking
Essential for real-time comparative analysis.
3. Streamlit (UI Layer)
Powers the web interface:
File uploads
Sliders, buttons, toggles
Responsive image display
Session-state handling
Rapid experiment-driven development
No JavaScript required.
4. Custom Modules (src/)
preprocess.py — cropping, resizing, denoising, contrast normalization
align.py — feature-based alignment using AKAZE/SIFT + RANSAC
diffmaps.py — tonal differences, edge maps, binary masks, overlays
utils.py — color conversions + helper functions
All components are modular and extendable.
🔬 Image Processing Pipeline
Below is the exact pipeline applied to every pair of impressions.
1. Preprocessing
Purpose: standardize impressions to maximize comparability.
Steps:
Convert to grayscale
Resize based on user-selected max dimension
Normalize contrast (CLAHE or linear)
Light denoising to reduce scanner noise
Optional cropping (if enabled)
2. Image Alignment
Intaglio impressions rarely match pixel-to-pixel due to:
Slight rotation
Scanner skew
Paper expansion/warp
Uneven cropping
Photographing vs scanning differences
Alignment workflow:
Detect features (AKAZE or SIFT)
Match features
Filter matches using Lowe ratio test
Estimate homography via RANSAC
Warp target impression to align with the base
Achieves subpixel precision when successful.
3. Tonal Difference Map
Used to detect:
Plate wear (loss of tone)
Rebitten or strengthened lines
Ink distribution differences
State-level changes
Operations:
diff_raw = abs(base - target_aligned)
norm = normalize(diff_raw)
mask = threshold(diff_raw)
User-adjustable slider controls threshold sensitivity.
4. Line-Difference Map (Experimental)
Extract edges → compare linework:
edges_base   = Canny(base)
edges_target = Canny(target_aligned)
line_diff    = XOR(edges_base, edges_target)
Useful for pinpointing added or effaced burin/needle work.
5. Red–Cyan Color Overlay
A connoisseur-friendly composite:
Base impression → red-tinted grayscale
Target impression → cyan-tinted grayscale
Regions of alignment → neutral
Differences → glow red or cyan
Supports:
Hide/show each impression
Thresholded mask view
Inversion (photographic negative)
Mirrors digital compositing workflows used in major print rooms.
🧪 Advanced Features
Swap base/target
Threshold slider for mask refinement
Inversion (black ↔ white)
Hide Base / Hide Target for isolating impressions
Original vs preprocessed comparisons
All visualizations update live.
🚀 How to Run Locally
1. Clone the repo
git clone https://github.com/yourusername/intaglio-comparator.git
cd intaglio-comparator
2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Launch the app
streamlit run app/app.py
Then open:
👉 http://localhost:8501
Repository Layout
intaglio-state-comparator/
  README.md
  requirements.txt
  
  data/
    raw/         # original test images
    processed/   # optional: preprocessed or aligned outputs

  src/
    preprocess.py
    align.py
    diffmaps.py
    utils.py
    features.py
    cluster.py    # optional: experimental clustering for state prediction

  notebooks/
    experiments.ipynb

  app/
    app.py        # Streamlit interface
This is a research prototype, not production software. Parameters (thresholds, grid sizes, feature detectors) are intentionally exposed for tuning across print types, papers, and digitization conditions.
