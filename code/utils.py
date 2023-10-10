import gzip
import numpy as np


def read_mnist_image(path: str, image_height, image_width, num_images):
    with gzip.open(path, "r") as f:
        f.read(16)  # skip non-image information
        buf = f.read(image_height * image_width * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, image_width * image_height)
        return data


def read_mnist_label(path: str, num_labels):
    with gzip.open(path, "r") as f:
        f.read(8)
        buf = f.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        labels = np.eye(10)[labels]
        return labels
