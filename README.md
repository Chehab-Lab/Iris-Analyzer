# Iris Analyzer

A Streamlit app for batch analysis of near-infrared (NIR) iris images. It ports
the original FastAPI `analyze_iris_image` logic into an interactive,
scientifically-styled UI.

For each uploaded image the app:

1. Runs an initial `iris` pipeline pass to obtain the pupil segmentation mask.
2. Detects bright specular reflections **inside the pupil** and removes them via
   OpenCV Navier–Stokes inpainting.
3. Re-runs the IRIS pipeline on the cleaned image.
4. Estimates pupil & iris geometry and computes the
   **Iris-to-Pupil Ratio** `IPR = iris_radius / pupil_radius`.
5. Renders an annotated overlay (pupil circle, iris boundary, centers).

## Features

- Upload **one or many** images (`png, jpg, jpeg, bmp, tif, tiff`).
- Per-image overlay + metric tiles, a combined results table, and a summary band.
- One-click **ZIP export** containing:
  ```
  iris_analysis_YYYYMMDD_HHMMSS.zip
  ├── Images/
  │   ├── 01_name.png        # annotated overlay per image
  │   └── 02_name.png
  ├── results.csv            # IPR + geometry + status for every image
  └── README.txt
  ```
- Standalone `results.csv` download.

## Install

```bash
pip install -r requirements.txt
```

> `open-iris` provides the `import iris` package (Worldcoin IRIS pipeline). If a
> compatible build is unavailable for your platform, the UI still loads and
> reports the engine as *missing* in the sidebar.

## Run

```bash
streamlit run app.py
```

## Configuration (sidebar)

| Setting | Description |
| --- | --- |
| **Eye side** | `right` / `left`, passed to the IRIS pipeline. |
| **Max image dimension** | Large images are downscaled (preserving aspect ratio) before analysis. |

## Output columns (`results.csv`)

`index, image, status, error, ipr, pupil_x, pupil_y, pupil_radius,
iris_x, iris_y, iris_radius, width, height`
