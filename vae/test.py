from vae.vae_model import CombTrain_VQ_VAE_dataset
import torchvision.transforms as transforms
from PIL import Image

tensorize_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
if __name__ == '__main__':

    # dataset测试
    train_imgs_path = '../z_using_files/imgs/content_images/LXGWWenKaiGB-Light_train/'
    val_imgs_path = '../z_using_files/imgs/content_images/LXGWWenKaiGB-Light_val/'
    train_dataset = CombTrain_VQ_VAE_dataset(train_imgs_path, transform=tensorize_transform)
    # 3000 ../z_using_files/imgs/content_images/LXGWWenKaiGB-Light_train/丁.png
    # print(len(train_dataset.imgs), train_dataset.imgs[1])
    img = Image.open(train_dataset.imgs[10])
    img = train_dataset.transform(img)
    # torch.Size([1, 128, 128])
    print(img.shape)
    print(train_dataset[0].shape)
