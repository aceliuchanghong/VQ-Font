import os
import shutil
import random


def split_content_folders(
    input_directory, train_directory, val_directory, train_count=3000, val_count=500
):
    if not os.path.exists(train_directory):
        os.makedirs(train_directory)
    if not os.path.exists(val_directory):
        os.makedirs(val_directory)

    files_list = os.listdir(input_directory)

    random.shuffle(files_list)

    train_folders_imgs = files_list[:train_count]
    val_folders_imgs = files_list[train_count : train_count + val_count]

    for img in train_folders_imgs:
        src_path = os.path.join(input_directory, img)
        dst_path = os.path.join(train_directory, img)
        shutil.copy(src_path, dst_path)
        print(f"copy {src_path} to {dst_path}")

    for img in val_folders_imgs:
        src_path = os.path.join(input_directory, img)
        dst_path = os.path.join(val_directory, img)
        shutil.copy(src_path, dst_path)
        print(f"copy {src_path} to {dst_path}")


if __name__ == "__main__":
    """
    conda activate VQFont
    cd datasets
    python split_folder.py --input_directory ../z_using_files/f2p_imgs/Alibaba-PuHuiTi-Medium
    python split_folder.py --input_directory ../z_using_files/f2p_imgs/SourceHanSansCN-Medium
    """

    import argparse

    parser = argparse.ArgumentParser(
        description="Split folders into train and validation sets"
    )
    parser.add_argument("--input_directory", type=str, help="Path to input directory")
    args = parser.parse_args()

    input_directory = args.input_directory
    train_directory = input_directory + "_train"
    val_directory = input_directory + "_val"

    # split_content_folders(input_directory, train_directory, val_directory)
    split_content_folders(
        input_directory,
        train_directory,
        val_directory,
    )
