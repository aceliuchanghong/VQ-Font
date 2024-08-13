import sys

sys.path.append("../")
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from model import content_enc_builder
from model import dec_builder

from torch.utils.data import Dataset
from PIL import Image


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)

    def forward(self, inputs):
        # 传入的是图片经过encoder后的feature maps
        # convert inputs from BCHW
        input_shape = inputs.shape

        # Flatten input ->[BC HW]
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # 得到编号
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized, perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.999, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.0):
        super(Model, self).__init__()

        self._encoder = content_enc_builder(1, 32, 256)

        if decay > 0.0:
            # decay是指数移动平均（EMA）的衰减因子
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = dec_builder(32, 1)

    def forward(self, x):
        z = self._encoder(x)  # [B 256 16 16]
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        """
        loss:量化过程中的损失值，它包括两个部分：编码空间的均方误差（e_latent_loss）和解码空间的均方误差（q_latent_loss），
        同时还有一个由commitment_cost控制的权重参数。这是模型的总损失值
        
        quantized:量化后的特征图（feature map），即输入特征图经过量化操作后的输出。这个量化操作是通过查找表（embedding table）实现的
        
        x_recon:经过编码器提取特征、量化器量化特征、再通过解码器还原后的输出。它与原始输入x相比，应该是一个尽可能接近的重建结果。
        这个重建结果通常用于计算重建误差,是模型对输入数据的重建版本
        
        perplexity:perplexity 是困惑度，衡量编码分布的多样性。它计算了平均编码分布的熵。
        理想情况下，perplexity 的值应接近于模型的嵌入向量数量，表示模型有效地使用了所有嵌入向量
        perplexity 越接近于嵌入向量数量越好。太低表示模型没有充分利用嵌入向量，太高则可能意味着模型在特定嵌入上过度集中
        """
        return loss, x_recon, perplexity


class CombTrain_VQ_VAE_dataset(Dataset):
    """
    CombTrain_VQ_VAE_dataset, learn the laten codebook from content font.
    """

    def __init__(self, root, transform=None):
        self.img_path = root
        self.transform = transform
        self.imgs = self.read_file(self.img_path)
        # img = Image.open(self.imgs[0])
        # img = self.transform(img)
        # torch.Size([1, 128, 128])
        # print(img.shape)

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
        return img

    def __len__(self):
        return len(self.imgs)
