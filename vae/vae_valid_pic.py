import glob
import os
import sys

sys.path.append("../")
from torchvision.utils import make_grid
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import logging
from vae.vae_model import Model, CombTrain_VQ_VAE_dataset
from vae.vae_train import save_image

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def valid_model(val_imgs_path,
                embedding_dim=256,
                num_embeddings=100,
                commitment_cost=0.25,
                decay=0,
                model_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pic_name = 'best'
    model = Model(num_embeddings, embedding_dim, commitment_cost, decay).to(device)
    if model_path is not None:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model and optimizer from {model_path}")
        pic_name = os.path.basename(model_path).replace("VQ-VAE_chn_step", "S").replace(".pth", '')

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    logger.info('Validation started')
    tensorize_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    val_dataset = CombTrain_VQ_VAE_dataset(val_imgs_path, transform=tensorize_transform)
    validation_loader = DataLoader(val_dataset, batch_size=16, shuffle=True,
                                   drop_last=True, pin_memory=True)

    def validate_model(model, validation_loader):
        model.eval()
        valid_originals = next(iter(validation_loader))
        valid_originals = valid_originals.to(device)
        vq_output_eval = model.module._encoder(valid_originals) if hasattr(model, 'module') else model._encoder(
            valid_originals)
        _, valid_quantize, _, _ = model.module._vq_vae(vq_output_eval) if hasattr(model, 'module') else model._vq_vae(
            vq_output_eval)
        valid_reconstructions = model.module._decoder(valid_quantize) if hasattr(model, 'module') else model._decoder(
            valid_quantize)
        return valid_originals, valid_reconstructions

    org, recon_out = validate_model(model, validation_loader)
    save_image(make_grid((org + 0.5).cpu().data), f'../z_using_files/imgs/{pic_name}_00.png')
    save_image(make_grid((recon_out + 0.5).cpu().data), f'../z_using_files/imgs/{pic_name}_01.png')


if __name__ == '__main__':
    """
    cd vae
    python vae_valid_pic.py
    """
    val_imgs_path = '../z_using_files/imgs/content_images/LXGWWenKaiGB-Light_val/'
    model_path_base = '../weight/'
    model_files = glob.glob(os.path.join(model_path_base, 'VQ-VAE_chn_step_*.pth'))
    for model_path in model_files:
        print(model_path)
        valid_model(val_imgs_path, model_path=model_path)
