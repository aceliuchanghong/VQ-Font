# -*- coding: utf-8 -*-
import argparse
from ttf_utils import *


if __name__ == "__main__":
    """
    conda activate VQFont
    python datasets/f2p.py --font C:\\Users\\liuch\\Desktop\\new_font\\siyuanHT.otf
    python datasets/f2p.py --font C:\\Users\\liuch\\Desktop\\new_font\\torch_cpp.ttf
    python datasets/f2p.py --font z_using_files/content_font/LXGWWenKaiGB-Light.ttf --out z_using_files/f2p_imgs
    python datasets/f2p.py --font z_using_files/content_font/SourceHanSerifCN-Medium.ttf --out z_using_files/f2p_imgs
    python datasets/f2p.py --font z_using_files/content_font/Alibaba-PuHuiTi-Medium.ttf --out z_using_files/f2p_imgs
    python datasets/f2p.py --font z_using_files/content_font/SourceHanSansCN-Medium.otf --out z_using_files/f2p_imgs
    python datasets/f2p.py --font z_using_files/content_font/SourceHanSansCN-Medium.otf --out z_using_files/all_font_pics --char_file datasets/char_all_15000.txt --image_size 96
    mv z_using_files/imgs_2/VQ-VAE_chn_best-hp/*.png z_using_files/all_font_pics/SourceHanSansCN-Medium
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--font", required=True, help="font path")
    parser.add_argument("--out", required=True, help="img out path")
    parser.add_argument(
        "--char_file",
        default="datasets/Chinese_characters_3500.txt",
        help="char_file path",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
        help="Size of the image",
    )
    opt = parser.parse_args()
    font_path = opt.font

    with open(opt.char_file, "r") as f:
        char2img_list = f.read()
    if not opt.out:
        image_file_path = os.path.dirname(font_path)
    else:
        image_file_path = opt.out
    try:
        font2image(font_path, image_file_path, char2img_list, opt.image_size)
        print(os.path.basename(font_path), ":", image_file_path)
    except Exception as e:
        print(e)
