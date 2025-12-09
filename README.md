# **Intaglio State Comparator**  
### *A Computer Vision Tool for Comparative Analysis of Intaglio Print Impressions*

---

## **Overview**

The **Intaglio State Comparator** is a computer-vision research tool designed for **connoisseurship**, **technical art history**, and **print-room analysis**.

Given two digitized impressions of the *same intaglio plate* (etching, engraving, drypoint, mezzotint, etc.), the tool:

- Standardizes both images  
- Aligns the second impression to the first  
- Computes pixel-level difference maps  
- Generates interpretive overlays showing where linework, tone, or plate condition differ  
- Provides interactive controls for thresholding, inversion, visibility, and comparison direction  

**The tool does *not* assign states automatically** — it surfaces subtleties worth close looking.

### Optimized for:

- Comparing **different states** of a plate  
- Studying **plate wear**, rework, foul biting, drypoint burr  
- Identifying **strengthened or effaced lines**  
- Examining differences between impressions across collections  
- Supporting **cataloging**, scholarly research, and catalogues raisonnés  

---

# **Technology Stack**

The tool is written in **Python** using high-performance open-source imaging libraries.

---

## **Core Libraries**

### **1. OpenCV (`cv2`)**
Handles all major computer-vision operations:

- Image decoding (JPEG/PNG/TIFF → arrays)  
- Grayscale conversion  
- Resizing & normalization  
- Feature detection (AKAZE or SIFT)  
- Feature matching + Lowe ratio filtering  
- RANSAC homography estimation  
- Image warping & alignment  
- Edge extraction (Canny)  
- Difference-map computation  
- Color conversions (BGR/RGB/HSV)

---

### **2. NumPy**
Vectorized numerical operations for:

- Pixel arithmetic  
- Masking & thresholding  
- Red/cyan overlay tinting  
- Contrast normalization  
- Array manipulation  

---

### **3. Streamlit (UI Layer)**
Provides the interactive web interface:

- File uploads  
- Sliders, toggles, radio buttons  
- Real-time visualization  
- Session state  
- Responsive, browser-based display  

---

### **4. Custom Modules (`/src`)**

- **`preprocess.py`** — cropping, resizing, denoising, contrast normalization  
- **`align.py`** — feature-based alignment using AKAZE/SIFT + RANSAC  
- **`diffmaps.py`** — tonal difference maps, edge maps, overlays  
- **`utils.py`** — color conversions, helpers  
- **`features.py`** — grid-based feature extraction (optional)  
- **`cluster.py`** — experimental KMeans clustering for impression grouping  

---

# 🔬 **Image Processing Pipeline**

### **1. Preprocessing**
- Convert to grayscale  
- Resize to standardized dimensions  
- Normalize contrast (CLAHE or linear)  
- Light denoising  
- Optional cropping  

Purpose: improve comparability and reduce digitization noise.

---

### **2. Image Alignment**
Because impressions differ slightly due to scanner skew, paper distortion, or photography:

1. Detect features (AKAZE or SIFT)  
2. Match features  
3. Apply Lowe ratio test  
4. Estimate homography with RANSAC  
5. Warp target image to base image  

Produces **subpixel alignment** when successful.

---

### **3. Tonal Difference Map**
Highlights:

- Plate wear (loss of tone)  
- Rebitten or strengthened lines  
- Ink distribution changes  

