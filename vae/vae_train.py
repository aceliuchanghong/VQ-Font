from vae.vae_model import CombTrain_VQ_VAE_dataset, Model
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import optim
from six.moves import xrange
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model.modules import weights_init
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
num_training_updates = 50000
embedding_dim = 256
num_embeddings = 100
commitment_cost = 0.25
decay = 0
learning_rate = 2e-4
train_imgs_path = 'path/to/save/train_content_imgs/'
tensorize_transform = transforms.Compose([transforms.Resize((128, 128)),
                                          transforms.ToTensor()])

train_dataset = CombTrain_VQ_VAE_dataset(train_imgs_path, transform=tensorize_transform)

train_loader = DataLoader(train_dataset, batch_size=128, batch_sampler=None, drop_last=True, pin_memory=True,
                          shuffle=True)

model = Model(num_embeddings, embedding_dim, commitment_cost, decay).to(device)
model.apply(weights_init("xavier"))

optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

model.train()
train_res_recon_error = []
train_res_perplexity = []
train_vq_loss = []


def val(model, validation_loader):
    model.eval()

    valid_originals = next(iter(validation_loader))
    valid_originals = valid_originals.to(device)

    vq_output_eval = model._encoder(valid_originals)
    _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
    valid_reconstructions = model._decoder(valid_quantize)


def show(img):
    npimg = img.numpy()
    fig = plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)


for i in xrange(num_training_updates):
    data = next(iter(train_loader))
    train_data_variance = torch.var(data)
    # print(train_data_variance)
    # show(make_grid(data.cpu().data) )
    # break
    data = data - 0.5  # normalize to [-0.5, 0.5]
    data = data.to(device)
    optimizer.zero_grad()

    vq_loss, data_recon, perplexity = model(data)
    # data_recon重构图像
    # print("vq_loss\n",vq_loss)
    recon_error = F.mse_loss(data_recon, data) / train_data_variance
    loss = recon_error + vq_loss
    # 重构损失更新encoder以及decoder,vq_loss用来更新embedding空间
    loss.backward()

    optimizer.step()

    train_res_recon_error.append(recon_error.item())
    train_res_perplexity.append(perplexity.item())
    train_vq_loss.append(vq_loss.item())

    if (i + 1) % 1000 == 0:
        print('%d iterations' % (i + 1))
        print('recon_error: %.3f' % np.mean(train_res_recon_error[-1000:]))
        print('perplexity: %.3f' % np.mean(train_res_perplexity[-1000:]))
        print('vq_loss: %.3f' % np.mean(train_vq_loss[-1000:]))
        print()
