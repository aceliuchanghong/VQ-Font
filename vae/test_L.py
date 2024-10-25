import torch
import torch.nn.functional as F
import lightning as L
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim
import logging
import os
from dotenv import load_dotenv
import sys

sys.path.append("../")

from model.content_encoder import content_enc_builder
from model.decoder import dec_builder
from vae.vae_model import (
    CombTrain_VQ_VAE_dataset,
    VectorQuantizer,
    VectorQuantizerEMA,
)

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
# ------------------------------
# Step 1: Define a LightningModule
# ------------------------------


class LitVQVAE(L.LightningModule):
    def __init__(
        self, num_embeddings=100, embedding_dim=256, commitment_cost=0.25, decay=0.0
    ):
        super().__init__()

        self._encoder = content_enc_builder(1, 32, 256)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(
                num_embeddings, embedding_dim, commitment_cost, decay
            )
        else:
            self._vq_vae = VectorQuantizer(
                num_embeddings, embedding_dim, commitment_cost
            )
        self._decoder = dec_builder(32, 1)

    def forward(self, x):
        valid_originals = x
        vq_output_eval = self._encoder(valid_originals)

        _, valid_quantize, _, _ = self._vq_vae(vq_output_eval)

        valid_reconstructions = self._decoder(valid_quantize)

        return valid_reconstructions

    def training_step(self, batch, batch_idx):

        data = batch
        data = data - 0.5  # 归一化到[-0.5, 0.5]
        x = data
        train_data_variance = torch.var(data)

        z = self._encoder(x)
        vq_loss, quantized, perplexity, _ = self._vq_vae(z)
        data_recon = self._decoder(quantized)

        recon_error = F.mse_loss(data_recon, data) / train_data_variance
        loss = recon_error.mean() + vq_loss.mean()

        logger.info(f"train_loss:{loss}")
        logger.info(f"recon_error:{recon_error.mean()}")
        logger.info(f"perplexity:{perplexity.mean()}")
        logger.info(f"vq_loss:{vq_loss.mean()}")

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=2e-4)
        return optimizer


# -------------------
# Step 2: Define data
# -------------------


def prepare_data(train_imgs_path, batch_size=1536):
    transform = transforms.Compose(
        [transforms.Resize((128, 128)), transforms.ToTensor()]
    )
    train_dataset = CombTrain_VQ_VAE_dataset(train_imgs_path, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )
    return train_loader


# -------------------
# Step 3: Train
# -------------------


def main():
    train_imgs_path = "../z_using_files/f2p_imgs/Alibaba-PuHuiTi-Medium_train/"

    # 配置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Training started")

    train_loader = prepare_data(train_imgs_path, batch_size=1536)

    model = LitVQVAE(
        num_embeddings=100, embedding_dim=256, commitment_cost=0.25, decay=0.999
    )

    trainer = L.Trainer(max_epochs=5000)
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    # cd vae
    # python test.py
    main()
