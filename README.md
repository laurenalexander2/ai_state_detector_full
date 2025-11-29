
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
