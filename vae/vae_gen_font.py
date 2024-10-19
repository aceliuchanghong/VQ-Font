import os
import sys

sys.path.append("../")
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import logging
from vae.vae_model import Model, CombTrain_VQ_VAE_dataset
from vae.vae_train import save_image

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 修改保存图片的部分，使每个张量单独保存为图片
def save_images_per_tensor(images, pic_name, save_path='../z_using_files/imgs/'):
    os.makedirs(save_path, exist_ok=True)  # 确保保存路径存在
    for idx, image in enumerate(images):
        image_path = os.path.join(save_path, f'{pic_name}_recon_{idx}.png')
        save_image(image, image_path)
        print(f'Saved image {idx} at {image_path}')


def validate_model(model, validation_loader):
    model.eval()
    all_originals = []
    all_reconstructions = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 遍历整个 validation_loader，逐批处理
    for batch in validation_loader:
        batch = batch.to(device)
        # 编码并量化每个批次
        vq_output_eval = model._encoder(batch)
        _, quantize, _, _ =  model._vq_vae(
            vq_output_eval)
        # 解码每个批次
        reconstructions =  model._decoder(quantize)

        # 收集原始图像和重构图像
        all_originals.append(batch)
        all_reconstructions.append(reconstructions)

    # 将所有批次的输出合并为单个张量
    all_originals = torch.cat(all_originals, dim=0)
    all_reconstructions = torch.cat(all_reconstructions, dim=0)

    return all_originals, all_reconstructions


def valid_model(val_imgs_path,
                embedding_dim=256,
                num_embeddings=100,
                commitment_cost=0.25,
                decay=0.0,
                model_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pic_name = 'best'
    model = Model(num_embeddings, embedding_dim, commitment_cost, decay).to(device)
    if model_path is not None:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model and optimizer from {model_path}")
        pic_name = os.path.basename(model_path).replace("VQ-VAE_chn_step", "S").replace(".pth", '')

    logger.info('Validation started')
    tensorize_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    val_dataset = CombTrain_VQ_VAE_dataset(val_imgs_path, transform=tensorize_transform)
    validation_loader = DataLoader(val_dataset, batch_size=16, shuffle=True,
                                   drop_last=True, pin_memory=True)

    org, recon_out = validate_model(model, validation_loader)
    save_images_per_tensor((recon_out + 0.5).cpu().data, pic_name)


if __name__ == '__main__':
    """
    cd vae
    python vae_valid_pic.py
    """
    val_imgs_path = '../z_using_files/imgs/content_images/LXGWWenKaiGB-Light_val/'
    model_path = '../weight/VQ-VAE_chn_step_800.pth'

    valid_model(val_imgs_path, model_path=model_path, decay=0.999)
