import sys

sys.path.append("../")
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model.modules import weights_init
import logging
from vae.vae_model import Model, CombTrain_VQ_VAE_dataset
import pandas as pd

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(
    train_imgs_path,
    val_imgs_path,
    num_training_updates=10000,
    embedding_dim=256,
    num_embeddings=100,
    commitment_cost=0.25,
    decay=0.0,
    learning_rate=2e-4,
    batch_size=512,
    model_path=None,
):
    # 确定设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 定义数据预处理
    tensorize_transform = transforms.Compose(
        [transforms.Resize((128, 128)), transforms.ToTensor()]
    )

    # 创建训练数据集和数据加载器
    train_dataset = CombTrain_VQ_VAE_dataset(
        train_imgs_path, transform=tensorize_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )

    # 初始化模型
    model = Model(num_embeddings, embedding_dim, commitment_cost, decay).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, capturable=True)

    # 加载模型和优化器状态
    if model_path is not None:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Loaded model and optimizer from {model_path}")
    else:
        model.apply(weights_init("xavier"))

    # multi-GPU support
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    logger.info("Training started")
    model.train()
    train_res_recon_error = []
    train_res_perplexity = []
    train_vq_loss = []
    steps_record = []

    # 创建一个DataFrame用于存储每一步的损失值
    loss_records_df = pd.DataFrame(
        columns=["step", "recon_error", "perplexity", "vq_loss"]
    )
    excel_file_path = f"../weight/vae_training_loss_{num_training_updates}.xlsx"

    # 训练循环
    for i in range(num_training_updates):
        data = next(iter(train_loader))
        train_data_variance = torch.var(data)

        data = data - 0.5  # 归一化到[-0.5, 0.5]
        data = data.to(device)
        optimizer.zero_grad()

        vq_loss, data_recon, perplexity = model(data)
        recon_error = F.mse_loss(data_recon, data) / train_data_variance
        loss = recon_error.mean() + vq_loss.mean()  # 计算平均损失
        loss.backward()

        optimizer.step()

        train_res_recon_error.append(recon_error.mean().item())
        train_res_perplexity.append(perplexity.mean().item())
        train_vq_loss.append(vq_loss.mean().item())
        steps_record.append(i + 1)

        if (i + 1) % 10 == 0:
            logger.info(f"{i + 1} iterations")
            logger.info(f"recon_error: {np.mean(train_res_recon_error[-10:]):.3f}")
            logger.info(f"perplexity: {np.mean(train_res_perplexity[-10:]):.3f}")
            logger.info(f"vq_loss: {np.mean(train_vq_loss[-10:]):.3f}")
            print("")

            # 将当前步的损失值记录到DataFrame中
            loss_record = {
                "step": i + 1,
                "recon_error": np.mean(train_res_recon_error[-10:]),
                "perplexity": np.mean(train_res_perplexity[-10:]),
                "vq_loss": np.mean(train_vq_loss[-10:]),
            }
            record_df = pd.DataFrame([loss_record])
            loss_records_df = pd.concat([loss_records_df, record_df], ignore_index=True)
            # 将DataFrame写入到Excel文件中
            loss_records_df.to_excel(excel_file_path, index=False)

        if (i + 1) % 100 == 0:
            save_checkpoint(model, optimizer, f"../weight/VQ-VAE_chn_step_{i + 1}.pth")

    save_checkpoint(model, optimizer, "../weight/VQ-VAE_chn_last.pth")

    # 验证部分
    logger.info("Validation started")
    val_dataset = CombTrain_VQ_VAE_dataset(val_imgs_path, transform=tensorize_transform)
    validation_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=True, drop_last=True, pin_memory=True
    )

    def validate_model(model, validation_loader):
        model.eval()
        valid_originals = next(iter(validation_loader))
        valid_originals = valid_originals.to(device)
        vq_output_eval = (
            model.module._encoder(valid_originals)
            if hasattr(model, "module")
            else model._encoder(valid_originals)
        )
        _, valid_quantize, _, _ = (
            model.module._vq_vae(vq_output_eval)
            if hasattr(model, "module")
            else model._vq_vae(vq_output_eval)
        )
        valid_reconstructions = (
            model.module._decoder(valid_quantize)
            if hasattr(model, "module")
            else model._decoder(valid_quantize)
        )
        return valid_originals, valid_reconstructions

    original, reconstruct = validate_model(model, validation_loader)
    save_image(make_grid((original + 0.5).cpu().data), "../z_using_files/imgs/00.png")
    save_image(
        make_grid((reconstruct + 0.5).cpu().data), "../z_using_files/imgs/01.png"
    )


def save_checkpoint(model, optimizer, filepath):
    state = {
        "model_state_dict": (
            model.module.state_dict()
            if hasattr(model, "module")
            else model.state_dict()
        ),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state, filepath)


def save_image(img, filepath):
    npimg = img.numpy()
    npimg = np.clip(npimg, 0, 1)  # Ensure values are in the 0..1 range
    plt.imsave(filepath, np.transpose(npimg, (1, 2, 0)))


def main():
    # train_imgs_path = '../z_using_files/f2p_imgs/LXGWWenKaiGB-Light_train/'
    # val_imgs_path = '../z_using_files/f2p_imgs/LXGWWenKaiGB-Light_val/'

    train_imgs_path = "../z_using_files/f2p_imgs/Alibaba-PuHuiTi-Medium_train/"
    val_imgs_path = "../z_using_files/f2p_imgs/Alibaba-PuHuiTi-Medium_val/"

    # model_path = '../weight/VQ-VAE_chn_best.pth'
    model_path = None
    train_model(
        train_imgs_path,
        val_imgs_path,
        num_training_updates=40000,
        batch_size=1536,
        model_path=model_path,
        decay=0.999,
    )


if __name__ == "__main__":
    # cd vae
    # python vae_train.py
    # nohup python vae_train.py>train.log &
    main()
