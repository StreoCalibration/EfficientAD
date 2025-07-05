import numpy as np
import cv2

def generate_normal_image(image_size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Generates a synthetic normal image (a plain gray square).

    Args:
        image_size (tuple[int, int]): The (height, width) of the image to generate.

    Returns:
        np.ndarray: The generated normal image.
    """
    # Create a gray image
    image = np.full((*image_size, 3), 128, dtype=np.uint8)
    return image

def generate_anomalous_image(image_size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Generates a synthetic anomalous image (a gray square with a white rectangle).

    Args:
        image_size (tuple[int, int]): The (height, width) of the image to generate.

    Returns:
        np.ndarray: The generated anomalous image.
    """
    image = generate_normal_image(image_size)
    
    # Add a random white rectangle as an anomaly
    h, w = image_size
    x1 = np.random.randint(0, w - 20)
    y1 = np.random.randint(0, h - 20)
    x2 = x1 + np.random.randint(10, 20)
    y2 = y1 + np.random.randint(10, 20)
    
    # Draw a filled white rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)
    
    return image
