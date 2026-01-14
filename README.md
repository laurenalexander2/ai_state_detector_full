# **Print Comparator**  
### *A Computer Vision Tool for Comparative Analysis of Print Impressions*
**Try it out:** https://matrixvision.streamlit.app/
**Download and run locally:** streamlit run app/app.py
---

## **Overview**

The **Print Comparator** is a computer-vision research tool designed for **technical art history** and **print-room analysis**.

Given two digitized impressions of the *same plate*, the tool:

- Standardizes both images  
- Aligns the second impression to the first  
- Computes pixel-level difference maps  
- Generates interpretive overlays showing where linework, tone, or plate condition differ  
- Provides interactive controls for thresholding, inversion, visibility, and comparison direction  

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

# **Image Processing Pipeline**

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
After alignment, the base image and warped target are compared using a per-pixel absolute intensity difference:
- abs(base_gray − aligned_target)

- Optional normalization for visualization
- Optional global thresholding to produce a binary mask
- This representation emphasizes low-frequency tonal divergence rather than discrete edge changes, making it sensitive to:
- Plate wear and burnishing
- Inking density variation
- Retroussage and wiping patterns
- Broad changes in plate tone

The binary mask is derived via a fixed or user-adjustable threshold and is primarily intended as a region-of-interest extractor rather than a semantic classifier.

---

### ***4. Color Overlay (Channel-Mapped Composite)***
The color overlay maps each aligned impression into separate color channels:
- Base → Red
- Target → Cyan (Green + Blue)

- The composite is formed by additive channel blending after normalization.
- No edge detection or thresholding is applied at this stage.
- Interpretive behavior:
- Spatial agreement → neutral gray
- Line displacement, reinforcement, or loss → chromatic separation
- Subtle mismatches remain visible without hard binarization

This mode preserves continuous spatial information and is therefore preferred for close visual inspection of line structure and cross-hatching.
