# -*- coding: utf-8 -*-
import os
import fontforge
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def draw_glyph(font_path, char_dict):
    try:
        try:
            font = fontforge.open(font_path)
        except Exception as e:
            print(f"Error open {font_path}: {e}")
        font.em = 256
        # 输出图片路径=字体名称+生成数量
        output_subdir = os.path.join(
            output_dir, os.path.basename(font_path).split(".")[0] + str(len(char_dict))
        )
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        for char in char_dict:
            logger.info(f"generating: {char}")
            try:
                glyph = font[ord(char)]  # Get the glyph for the character
                glyph.export(os.path.join(output_subdir, f"{char}.png"), 127)  # 128*128
            except Exception as e:
                logger.error(f"Glyph not found for character {char}: {e}")

    except Exception as e:
        logger.error(f"Error processing font {font_path}: {e}")


if __name__ == "__main__":
    """
    apt-get install python3-fontforge
    cd datasets
    /usr/bin/python3 gen_imgs_from_ttf.py --test_ttf ../z_using_files/content_font/SourceHanSansCN-Medium.otf
    /usr/bin/python3 gen_imgs_from_ttf.py --test_ttf ../z_using_files/content_font/LXGWWenKaiGB-Light.ttf
    ffpython D:\\aProject\\py\\SDT\\z_new_start\\generate_utils\\gen_imgs_from_ttf.py
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="../z_using_files/all_font_pics",
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--test_ttf",
        default="../z_using_files/content_font/SourceHanSansCN-Medium.otf",
        help="Path to the TTF file to use",
    )
    parser.add_argument(
        "--char_file",
        default="char_all_15000.txt",
        help="Path char_file",
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    test_ttf = args.test_ttf
    with open(args.char_file, "r") as f:
        cha2img_list_new = f.read()

    char_dict = list(cha2img_list_new)

    try:
        draw_glyph(test_ttf, char_dict)
    except Exception as e:
        print(f"Error01: {e}")
