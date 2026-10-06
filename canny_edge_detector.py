"""
Implementation of the Canny edge detector
"""

import cv2
import numpy as np


def convolve(image, kernel, axis):
    """
    Convolve a 2D image with a 1D kernel along a specified axis
      axis = 0 -> vertical (neighbors above/below, same column)
      axis = 1 -> horizontal (neighbors left/right, same row)
    """

    height, width = image.shape
    kernel_length = len(kernel)
    kernel_radius = kernel_length // 2

    # Pad the borders
    # Pixels near the edge need neighbors that don't exist
    if axis == 0:
        padding = [(kernel_radius, kernel_radius), (0, 0)]
    else:
        padding = [(0, 0), (kernel_radius, kernel_radius)]

    padded = np.pad(image, padding, mode='reflect')

    # Flip the kernel to perform convolution
    flipped_kernel = kernel[ : : -1]

    result = np.zeros((height, width), dtype=np.float64)

    for k in range(kernel_length):
        weight = flipped_kernel[k]

        if axis == 0:
            shifted_image = padded[k : k + height, :]
        else:
            shifted_image = padded[:, k : k + width]

        # For each pixel, store weight * neighbor at this offset
        result += weight * shifted_image

    return result


def bilinear(image, y, x):
    """
    Approximates the value at a certain point in a grid
    Samples 4 coordinates with known values
    Takes a weighted average of the 4 surrounding points
    """

    height, width = image.shape

    # Known points: (x0, y0), (x0, y1), (x1, y0), (x1, y1)
    x0 = np.floor(x).astype(np.intp)
    y0 = np.floor(y).astype(np.intp)
    x1 = min(x0 + 1, width -1)
    y1 = min(y0 + 1, height - 1)

    right_weight = x - x0
    bottom_weight = y - y0
    left_weight = 1 - right_weight
    top_weight = 1 - bottom_weight

    return (
        top_weight * left_weight * image[y0, x0]
        + top_weight * right_weight * image[y0, x1]
        + bottom_weight * left_weight * image[y1, x0]
        + bottom_weight * right_weight * image[y1, x1]
    )


def non_maximum_suppression(M, theta):
    """
    Keep local maxima along the gradient
    Leave borders zero
    """

    height, width = M.shape

    # Initialized with all pixels suppressed
    suppressed = np.zeros_like(M)

    # Skip borders to compare within the image
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Convert angle into horizontal and vertical direction offsets
            dx = np.cos(theta[y, x])
            dy = np.sin(theta[y, x])

            # Extend the direction until its largest component is a single pixel
            scale = max(abs(dx), abs(dy))
            dx /= scale
            dy /= scale

            # Estimate magnitudes on both sides along the gradient direction
            # Bilinear interpolation handles points between point centers
            forward = bilinear(M, y + dy, x + dx)
            backward = bilinear(M, y - dy, x - dx)

            # Keep the center only if it is stronger than both comparisons
            if M[y, x] >= forward and M[y, x] > backward:
                suppressed[y, x] = M[y, x]

    return suppressed


def hysteresis(N, low, high):
    """
    Keep 8-connected candidate regions containing a strong pixel
    N is magnitude image after non-maximum suppression
    low and high define the threshold to consider a pixel an edge pixel
    """

    if 0 < low < high:
        # Select possible edge pixels, at or above lower bound
        candidates = (N >= low).astype(np.uint8)

        # Assign each connected region a label, label 0 is reserved for background
        count, labels = cv2.connectedComponents(candidates, connectivity=8)

        # Identify regions containing at least one strong pixel
        strong_labels = np.unique(labels[N >= high])

        # Create a lookup table of accepted groups
        keep = np.zeros(count, dtype=bool)
        keep[strong_labels] = True

        # Map retained region labels back to a binary edge image
        return keep[labels]
    
    else:
        raise ValueError("Requires 0 < low < high")


def read_image(path):
    """
    Read an image as a floating-point grayscale intensity matrix
    """

    I = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if I is None:
        raise FileNotFoundError(f'Unable to read image at "{path}"')
    
    return I.astype(np.float64)


def gaussian_masks(sigma):
    """
    Sample a normalized Gaussian and its first derivative
    Assuming a fininte and positive sigma value
    """

    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    G = np.exp(-x**2 / (2 * sigma**2))
    # Preserve constant intensities during smoothing
    G /= G.sum()
    # Same derivative coefficients for either axis
    dG = -(x / sigma**2) * G

    return G, dG


def canny(I, sigma=1.0, high_ratio=0.20, low_ratio=0.50):
    """
    Implementation of the Canny edge detector algorithm
    high_ratio multiplies the maximum suppressed magnitude
    low_ratio multiplies the high threshold
    """

    I = np.asarray(I, dtype=np.float64)
    G, dG = gaussian_masks(sigma)
    
    Ix = convolve(I, G, axis=0)
    Iy = convolve(I, G, axis=1)
    Ix_prime = convolve(Ix, dG, axis=1)
    Iy_prime = convolve(Iy, dG, axis=0)

    M = np.hypot(Ix_prime, Iy_prime)
    theta = np.arctan2(Iy_prime, Ix_prime)
    N = non_maximum_suppression(M, theta)

    high = high_ratio * N.max()
    low = low_ratio * high

    # Avoid zero thresholds for images with no remaining responses
    if high > 0:
        edges = hysteresis(N, low, high)
    else:
        edges = np.zeros_like(N, dtype=bool)

    return dict(I=I, sigma=sigma, G=G, dG=dG, Ix=Ix, Iy=Iy,
                Ix_prime=Ix_prime, Iy_prime=Iy_prime, M=M, theta=theta,
                N=N, edges=edges, low=low, high=high)
