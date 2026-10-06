"""
Generates stage images, stage grids, and sigma comparisons for the Canny detector
Paths are relative to this file
The detector lives is implemented in canny_edge_detector.py
"""
 
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from canny_edge_detector import canny, read_image
 
# (result key, saved filename, plot title)
stages = [
    ('I', 'original.png','Original'),
    ('Ix','(a)_gaussian_Ix.png', '(a) Gaussian component Ix\n(vertical smoothing)'),
    ('Iy', '(b)_gaussian_Iy.png', '(b) Gaussian component Iy\n(horizontal smoothing)'),
    ('Ix_prime', '(c)_x_derivative.png', '(c) X derivative of Gaussian'),
    ('Iy_prime', '(d)_y_derivative.png', '(d) Y derivative of Gaussian'),
    ('M', '(e)_magnitude.png',  '(e) Gradient magnitude'),
    ('N', '(f)_non_max_suppression.png', '(f) After non-maximum suppression'),
    ('edges', 'final_edges.png', 'Final edges'),
]

grid_stages = stages[1:7]
 
 
def display_range(key, result, magnitude_max=None):
    """
    Return (black, white) values for a stage
    """

    if key in ('Ix_prime', 'Iy_prime'):
        # Signed: -max is black, 0 is gray, +max is white
        biggest = max(np.abs(result['Ix_prime']).max(),
                      np.abs(result['Iy_prime']).max()) or 1
        return -biggest, biggest
    
    if key in ('M', 'N'):
        return 0, (magnitude_max or result['M'].max()) or 1
    
    if key == 'edges':
        return 0, 1

    # I, Ix, Iy
    return 0, 255
 
 
def save_stage_images(result, folder):
    """
    Save each stage as its own PNG
    """

    folder.mkdir(parents=True, exist_ok=True)

    for key, filename, _ in stages:

        lower, upper = display_range(key, result)

        # Map [lower, upper] to [0, 1], clip, then convert
        scaled = np.clip((result[key].astype(float) - lower) / (upper - lower), 0, 1)

        pixels = np.round(scaled * 255).astype(np.uint8)

        if not cv2.imwrite(str(folder / filename), pixels):
            raise OSError(f'Unable to save {folder / filename}')
        
    print(f'Saved individual stages to {folder}')
 
 
def plot_intermediates(result, image_name):
    """
    Arrange the intermediate Canny steps in a 2x3 grid
    """

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout='constrained')

    for ax, (key, _, title) in zip(axes.flat, grid_stages):

        lower, upper = display_range(key, result)

        ax.imshow(result[key], cmap='gray', vmin=lower, vmax=upper)
        ax.set_title(title)
        ax.axis('off')

    fig.suptitle(f"{image_name} | Intermediate results | sigma = {result['sigma']:g}")

    return fig
 
 
def plot_sigma_comparison(results, image_name):
    """Compare magnitude, suppression, and final edges across sigmas."""
    fig, axes = plt.subplots(len(results), 3, figsize=(12, 4 * len(results)),
                             squeeze=False, layout='constrained')
    
    # One shared magnitude scale so brightness is comparable across sigmas
    shared_max = max(r['M'].max() for r in results)
    columns = [('M', 'Magnitude'), ('N', 'Non-max suppression'), ('edges', 'Final edges')]
 
    for row, result in enumerate(results):
        for col, (key, title) in enumerate(columns):

            ax = axes[row, col]

            lower, upper = display_range(key, result, magnitude_max=shared_max)

            ax.imshow(result[key], cmap='gray', vmin=lower, vmax=upper)
            ax.set_title(f"sigma = {result['sigma']:g} | {title}")
            ax.axis('off')
                
    fig.suptitle(f"{image_name} | Effect of sigma")
    
    return fig
 
 
def main():
    root = Path(__file__).resolve().parent
    output = root / 'output'
 
    for image_name in ('elk', 'flowers', 'umbrellas'):
        I = read_image(root / 'images' / f'{image_name}.jpg')
        results = []
        for sigma in (0.6, 1.0, 1.15):
            result = canny(I, sigma)
            results.append(result)
            folder = output / image_name / f'sigma_{sigma:g}'
            save_stage_images(result, folder)
            plot_intermediates(result, image_name).savefig(folder / 'intermediates.png', dpi=180)
 
        plot_sigma_comparison(results, image_name).savefig(
            output / image_name / 'sigma_comparison.png', dpi=180)
        
        plt.close('all')
 
 
if __name__ == '__main__':
    main()