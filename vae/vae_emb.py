import os
import sys

sys.path.append("../")
import json
import torch
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import logging
from vae.vae_model import Model

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 模型参数
embedding_dim = 256
num_embeddings = 100
commitment_cost = 0.25
decay = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = Model(num_embeddings, embedding_dim, commitment_cost, decay).to(device)
models = torch.load('../weight/VQ-VAE_chn_.pth')
encoder = models._encoder
encoder.requires_grad = False
encoder.to("cpu")


# 自定义数据集
class CombTrain_VQ_VAE_dataset(Dataset):
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
        img = Image.open(img_name)
        if self.transform is not None:
            img = self.transform(img)  # 转换为 Tensor [C, H, W]
        return img_name, img

    def __len__(self):
        return len(self.imgs)


# 余弦相似度计算函数
def CosineSimilarity(tensor_1, tensor_2):
    normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
    normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
    return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)


def main(train_imgs_path, batch_size, output_path):
    # 数据转换和加载
    tensorize_transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    sim_dataset = CombTrain_VQ_VAE_dataset(train_imgs_path, transform=tensorize_transform)
    sim_loader = DataLoader(sim_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    similarity = []

    for data in sim_loader:
        img_name, img_tensor = data
        img_tensor = img_tensor - 0.5  # 归一化到 [-0.5, 0.5]
        img_tensor = img_tensor.to("cpu")

        # 获取内容特征
        content_feature = encoder(img_tensor)
        vector = content_feature.view(content_feature.shape[0], -1)

        sim_all = {}
        for i in range(batch_size):
            char_i = hex(ord(img_name[i][-5]))[2:].upper()
            dict_sim_i = {char_i: {}}
            for j in range(batch_size):
                char_j = hex(ord(img_name[j][-5]))[2:].upper()
                sim = CosineSimilarity(vector[i], vector[j])
                if i == j:
                    sim = 1.0
                dict_sim_i[char_i][char_j] = float(sim)
            sim_all.update(dict_sim_i)

        dict_json = json.dumps(sim_all)

        # 保存为.json文件
        with open(output_path, 'w+') as file:
            file.write(dict_json)

    # 读取并打印内容
    with open(output_path, 'r+') as file:
        content = json.load(file)
    print(content['4E08'])


if __name__ == "__main__":
    train_imgs_path = 'path/to/save/all_content_imgs'  # 图片路径
    batch_size = 3500  # 批处理大小
    output_path = '../weight/all_char_similarity_unicode.json'  # 输出文件路径
    main(train_imgs_path, batch_size, output_path)
