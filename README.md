# Canny Edge Detection

A step-by-step Python implementation of Canny edge detection. The project focuses on understanding the algorithm and inspecting its intermediate results, using NumPy for the calculations, OpenCV for image loading and connected components, and Matplotlib for plotting.

Gaussian filtering, derivative filtering, bilinear interpolation, and non-maximum suppression are implemented directly. The detector does not call OpenCV's `Canny` or Gaussian blur functions.

## How it works

1. Load a grayscale image as a floating-point matrix.
2. Sample a normalized 1D Gaussian and its first derivative, truncating the masks at approximately ±3σ.
3. Apply separable convolution with reflected borders to compute horizontal and vertical gradients.
4. Calculate gradient magnitude and orientation.
5. Thin responses with non-maximum suppression, comparing interpolated magnitudes on both sides along the gradient direction.
6. Apply hysteresis: retain 8-connected candidate regions containing at least one strong pixel.

By default, the high threshold is 20% of the maximum magnitude remaining after suppression. The low threshold is half the high threshold. These thresholds are recalculated for each run.

## Getting started

From the project folder, create a virtual environment and install the three dependencies needed by the scripts:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy opencv-python matplotlib
python run.py
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. `requirements.txt` records the larger development environment, including Jupyter packages; it can be installed with `python -m pip install -r requirements.txt`.

The runner processes `elk.jpg`, `flowers.jpg`, and `umbrellas.jpg` at σ values **0.6, 1.0, and 1.15**. It saves the results automatically and closes the figures without opening plot windows.

## Project files

| File or folder | Purpose |
| --- | --- |
| `canny_edge_detector.py` | Reusable filtering, interpolation, suppression, hysteresis, and complete Canny pipeline. |
| `run.py` | Runs the image/sigma experiments and saves individual images and comparison figures. |
| `canny-detection.ipynb` | Step-by-step exploration of the implementation. |
| `images/` | Three sample images from the Berkeley Segmentation Dataset, obtained through [this dataset listing](https://www.kaggle.com/datasets/adheshgarg/bsds300?resource=download). |
| `output/` | Generated results; excluded from Git and recreated by running the script. |

## Saved results

Each image has its own folder, with one subfolder per sigma value:

```text
output/
└── elk/
    ├── sigma_0.6/
    ├── sigma_1/
    ├── sigma_1.15/
    └── sigma_comparison.png
```

The same structure is created for flowers and umbrellas. Each sigma folder contains:

- `original.png`
- `(a)_gaussian_Ix.png` and `(b)_gaussian_Iy.png`
- `(c)_x_derivative.png` and `(d)_y_derivative.png`
- `(e)_magnitude.png`
- `(f)_non_max_suppression.png`
- `final_edges.png`
- `intermediates.png`, a 2×3 grid of stages (a)–(f)

`Ix` is the vertical smoothing pass used before the horizontal derivative; `Iy` is the horizontal smoothing pass used before the vertical derivative. Signed derivatives are displayed with zero as gray. Magnitude and edge images use black for zero. Saved PNGs are display-scaled versions of the arrays, not raw floating-point data.

## Using the detector directly

```python
from canny_edge_detector import canny, read_image

I = read_image("images/elk.jpg")
result = canny(I, sigma=1.15)
edges = result["edges"]  # Boolean matrix: True marks an edge pixel.
```

The result dictionary also includes the smoothing passes, gradients, magnitude (`M`), orientation (`theta`), suppressed magnitude (`N`), and threshold values. Change the image names and sigma values in `run.py` to try other inputs. Larger sigma values smooth more strongly; the preferred value depends on which details matter for the image.

## License

The code is available under the [MIT License](LICENSE). Sample images retain their original source terms.
