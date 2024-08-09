import os
from PIL import Image
import argparse


def resize_images_in_directory(input_directory, output_directory, size):
    """
    Resize all PNG and JPG images in the specified input directory to the given output size and save them to the output directory.

    :param input_directory: Directory containing input images
    :param output_directory: Directory to save resized images
    :param size: Desired output size as a tuple (width, height)
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg')):
            img_path = os.path.join(input_directory, filename)
            with Image.open(img_path) as img:
                img = img.resize((size, size), Image.ANTIALIAS)
                img.save(os.path.join(output_directory, filename))


if __name__ == '__main__':
    """
    conda activate VQFont
    python datasets/pic_resize.py /path/to/input/images /path/to/output/images --size 128
    """
    parser = argparse.ArgumentParser(description="Resize PNG and JPG images in a directory.")
    parser.add_argument('input_directory', type=str, help="Path to the input directory containing images")
    parser.add_argument('output_directory', type=str, help="Path to the output directory to save resized images")
    parser.add_argument('--size', type=int, default=128,
                        help="Desired output size (width height), default is 128x128")
    args = parser.parse_args()
    resize_images_in_directory(args.input_directory, args.output_directory, args.size)
