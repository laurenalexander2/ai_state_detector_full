Intaglio State Comparator
A Computer Vision Tool for Comparative Analysis of Intaglio Print Impressions

Overview
The Intaglio State Comparator is a computer-vision research tool designed to support connoisseurship and technical art history.

Given two digitized impressions of the same intaglio plate (etching, engraving, drypoint, mezzotint, etc.), the app:
Standardizes both images

Aligns the second impression to the first

Computes pixel-level difference maps

Generates interpretive overlays highlighting where lines, tone, or plate condition differ

Allows interactive exploration via toggles, thresholding, inversion, and visibility controls

The app is optimized for:
comparing different states of a plate
studying plate wear, rework, foul biting, drypoint burr
identifying strengthened or effaced lines
examining differences between impressions from different museums
supporting cataloging and catalogues raisonnés
It does not assign states automatically — it surfaces areas worth close looking.
🏗️ Technology Stack
This tool is written in Python and built on standard, high-performance, open-source libraries used in scientific imaging.
Core Libraries
1. OpenCV (cv2)
Used for all heavy-lifting in imaging and computer vision:
image decoding (JPEG/PNG/TIFF → arrays)
grayscale conversion
resizing & normalization
feature detection and matching
homography estimation
warping and alignment
edge extraction
difference-map generation
masking & overlay creation
color space conversions (BGR/RGB/HSV)
OpenCV gives the tool fast, reliable, vectorized operations that scale to large museum-quality scans.
2. NumPy
Used for pixel-level mathematical operations:
channel stacking
tinting red/cyan overlays
normalization
mask operations
thresholding
arithmetic on aligned images
NumPy provides efficient array-based computation essential for real-time comparison.
3. Streamlit (UI Layer)
Provides the interactive web interface:
file uploads
sliders, radio buttons, toggles
image display with auto-resize
responsive layout
session state for swapping and display toggles
Streamlit allows rapid experimentation and deployment without requiring JavaScript.
4. Custom Modules
Located in src/:
preprocess.py — cropping, resizing, denoising, contrast normalization
align.py — feature-based alignment using OpenCV (AKAZE/SIFT) + RANSAC homography
diffmaps.py — tonal difference, edge-aware difference, binary masks, overlays
utils.py — BGR/RGB conversions and misc helpers
These pieces are modular, testable, and easy to extend.
🔬 Image Processing Pipeline — Detailed Explanation
Below is the exact pipeline applied to every pair of impressions.
1. Preprocessing
Preprocessing ensures both impressions are standardized before comparison:
Convert to grayscale
Resize based on user-selected max width
Normalize contrast (CLAHE or linear scaling)
Light denoising to reduce scanner noise
Optional cropping in your codebase (if implemented)
Goal: maximize comparability while minimizing noise.
2. Image Alignment
Alignment is critical. Two impressions of the same plate rarely match pixel-by-pixel:
slight rotation
scanner skew
stretch from paper expansion
uneven cropping
photographing vs scanning differences
Alignment Steps:
Detect local features (AKAZE or SIFT)
Match features across images
Filter matches via Lowe ratio test
Fit a homography using RANSAC
Warp the target image onto the base
If successful, impressions are aligned with subpixel precision.
3. Tonal Difference Map
Compute tonal differences:
diff_raw = abs(base - target_aligned)
norm_diff = normalize(diff_raw)
mask = threshold(diff_raw)
Use cases:
plate wear (lighter tones)
rebitten lines (darker)
ink distribution differences
A slider allows fine control of mask sensitivity.
4. Experimental Line-Difference Map
Extract edges:
edges_base = Canny(base)
edges_target = Canny(aligned_target)
line_diff = XOR(edges_base, edges_target)
This isolates changed linework.
5. Color Overlay (Red vs Cyan) with Proper Tinting
The most connoisseur-friendly visualization:
Base impression → red-tinted grayscale
Target impression → cyan-tinted grayscale
Areas where both align become neutral
Differences glow red or cyan
Hide/toggle each layer independently
Optional inversion gives a photographic negative effect
This style mirrors digital compositing techniques used in leading print rooms.
🧪 Advanced Features
Swap base & target: flips comparison direction
Black/white inversion: inverts luminance but preserves color channels
Hide Base / Hide Target: isolate impressions
Threshold slider: manually tune the difference mask
Display original scans and preprocessed scans
Everything updates dynamically.
🚀 How to Run Locally
1. Clone the repo
git clone https://github.com/yourusername/intaglio-comparator.git
cd intaglio-comparator
2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Run the app
streamlit run app/app.py
Visit:
👉 http://localhost:8501

# AI-Assisted State Detection for Intaglio Prints

This repository contains a prototype tool for aligning multiple impressions of the same intaglio print, 
computing visual difference maps, and (optionally) clustering impressions into probable plate states.

## Features

- Image preprocessing (grayscale, contrast normalization, resizing)
- Feature-based image alignment using ORB and homography estimation
- Edge-based difference maps and binary change masks
- Red-overlay visualizations that highlight regions of change
- Simple grid-based feature extraction per impression
- KMeans clustering for grouping impressions into probable states
- Minimal Streamlit app for upload, alignment, and visualization

## Quickstart

1. **Create a virtual environment** (recommended) and activate it.

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:

   ```bash
   streamlit run app/app.py
   ```

4. Upload two digitized impressions of the same intaglio print and inspect the highlighted differences.

## Layout

```text
ai_state_detector_full/
  README.md
  requirements.txt
  data/
    raw/         # put your original test images here
    processed/   # optional: for saving preprocessed/aligned images
  src/
    __init__.py
    preprocess.py
    align.py
    diffmaps.py
    features.py
    cluster.py
    utils.py
  notebooks/     # for experiments / analysis
  app/
    app.py       # Streamlit interface
```

This is a research prototype, not production software. Many parameters (thresholds, grid sizes, etc.)
are intentionally exposed and can be tuned for different kinds of prints and digitization conditions.
