import numpy as np
import cv2
from google.colab.patches import cv2_imshow

def highlight_object_outline(image_path, outline_color=(0, 0, 255), thickness=5, darken_factor=0.5):
    """
    Highlights the actual object's outline (not the rectangular boundary)
    for images with transparent backgrounds.

    :param image_path: Path to the image (must have alpha channel)
    :param outline_color: (B, G, R) color for outline
    :param thickness: Outline thickness in pixels
    :param darken_factor: How much to darken the object near edges
    :return: Image with outlined object (on transparent background)
    """
    # Load with alpha
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    if img.shape[2] < 4:
        raise ValueError("Image must have an alpha channel for transparency.")

    # Split channels
    b, g, r, a = cv2.split(img)

    # Threshold alpha to create a mask of visible regions
    mask = cv2.threshold(a, 1, 255, cv2.THRESH_BINARY)[1]

    # Find contours of the non-transparent region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Make a darkened version of the image for blending
    rgb = cv2.merge((b, g, r))
    darkened = (rgb * darken_factor).astype(np.uint8)

    # Copy to preserve transparency
    outlined_rgb = rgb.copy()

    # Draw the colored outline around detected shape(s)
    cv2.drawContours(outlined_rgb, contours, -1, outline_color, thickness)

    # Optional: slightly darken pixels near the contour for extra emphasis
    edge_mask = np.zeros_like(mask)
    cv2.drawContours(edge_mask, contours, -1, 255, thickness)

    # Create a color array with the same shape as the masked region
    color_array = np.full_like(outlined_rgb[edge_mask > 0], outline_color, dtype=np.uint8)

    outlined_rgb[edge_mask > 0] = cv2.addWeighted(
        outlined_rgb[edge_mask > 0], 0.5,
        color_array, 0.5, 0
    )

    # Merge back alpha
    outlined_img = cv2.merge((outlined_rgb, a))

    return outlined_img

# --- Example usage ---
outlined = highlight_object_outline(
    "1.png", outline_color=(0, 0, 255), thickness=10, darken_factor=0.7
)
cv2_imshow(outlined)
