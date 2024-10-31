import os
import sys
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import logging
from PIL import Image
from torchvision.utils import make_grid

sys.path.append("../")
from vae.vae_train import save_image
from vae.vae_model import Model


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class CombTrain_VQ_VAE_dataset_with_name(Dataset):

    def __init__(self, root, transform=None):
        self.img_path = root
        self.transform = transform
        self.imgs = self.read_file(self.img_path)

    def read_file(self, path):
        """从文件夹中读取数据"""
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

    def __getitem__(self, index):
        img_name = self.imgs[index]
        # print(img_name[-5:-4])  # 一..输出文字
        img = Image.open(img_name)
        if self.transform is not None:
            img = self.transform(img)  # Tensor [C H W] [1 128 128]
        return img, img_name[-5:-4]

    def __len__(self):
        return len(self.imgs)


def save_image_new(img, filepath, size=(96, 96)):
    # 调整图像尺寸为 96x96
    resize_transform = transforms.Resize(size)
    img = resize_transform(img)  # 调整大小

    npimg = img.numpy()
    npimg = np.clip(npimg, 0, 1)  # 确保数值在0..1范围内

    # 如果是灰度图像（1通道），去掉通道维度
    if npimg.shape[0] == 1:
        # print(f"first")
        npimg = npimg.squeeze(0)  # 移除通道维度（灰度图）
        plt.imsave(filepath, npimg, cmap="gray")  # 以灰度图保存
    else:
        plt.imsave(filepath, np.transpose(npimg, (1, 2, 0)))  # 保存为RGB图像


def validate_model(
    model, validation_loader, device, output_dir="../z_using_files/imgs_2", pic_name="0"
):
    model.eval()

    # 遍历 validation_loader 中的所有 batch
    for batch_idx, (valid_originals, names) in enumerate(validation_loader):
        valid_originals = valid_originals.to(device)
        logger.info(f"names:{names}")

        # 编码
        vq_output_eval = model._encoder(valid_originals)
        _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)

        # 解码
        valid_reconstructions = model._decoder(valid_quantize)

        # 保存当前batch的图片
        os.makedirs(f"{output_dir}/original", exist_ok=True)
        save_image(
            make_grid((valid_originals + 0.5).cpu().data),
            f"{output_dir}/original/{pic_name}_original_batch_{batch_idx}.png",
        )
        os.makedirs(f"{output_dir}/{pic_name}", exist_ok=True)
        for i, img in enumerate(valid_reconstructions):
            save_image_new(
                img.detach().cpu(), f"{output_dir}/{pic_name}/{names[i]}.png"
            )

        # 释放显存，避免累积
        del valid_originals, valid_quantize, valid_reconstructions
        torch.cuda.empty_cache()


def valid_model(
    val_imgs_path,
    embedding_dim=256,
    num_embeddings=100,
    commitment_cost=0.25,
    decay=0.0,
    model_path=None,
    batch_size=16,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pic_name = "best"
    model = Model(num_embeddings, embedding_dim, commitment_cost, decay).to(device)
    if model_path is not None:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model and optimizer from {model_path}")
        pic_name = (
            os.path.basename(model_path)
            .replace("VQ-VAE_chn_step", "S")
            .replace(".pth", "")
        )

    logger.info("Gen started")
    tensorize_transform = transforms.Compose(
        [transforms.Resize((128, 128)), transforms.ToTensor()]
    )
    val_dataset = CombTrain_VQ_VAE_dataset_with_name(
        val_imgs_path, transform=tensorize_transform
    )
    validation_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    validate_model(model, validation_loader, device, pic_name=pic_name)


if __name__ == "__main__":
    """
    cd vae
    python vae_gen_font.py --val_imgs_path ../z_using_files/f2p_imgs/SourceHanSansCN-Medium_train --model_path ../weight/VQ-VAE_chn_best-sy2.pth
    python vae_gen_font.py --val_imgs_path ../z_using_files/f2p_imgs/SourceHanSansCN-Medium_train --model_path ../weight/VQ-VAE_chn_best-hp.pth
    python vae_gen_font.py --val_imgs_path ../z_using_files/f2p_imgs/HYChaoCuSongJ_train --model_path ../weight/VQ-VAE_chn_best-hychao.pth
    """

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--val_imgs_path",
        type=str,
        default="../z_using_files/f2p_imgs/SourceHanSansCN-Medium_train",
        help="Path to validation images directory",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../weight/VQ-VAE_chn_best-sy2.pth",
        help="Path to the trained model file",
    )

    args = parser.parse_args()
    valid_model(args.val_imgs_path, model_path=args.model_path, decay=0.999)
