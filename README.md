Method is based on [Vector Quantization](https://arxiv.org/abs/1711.00937), so we named our FFG method **VQ-Font**.

Paper can be found
at ```./Paper_IMG/``` | [Arxiv](https://arxiv.org/abs/2309.00827)｜[CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Pan_Few_Shot_Font_Generation_Via_Transferring_Similarity_Guided_Global_Style_ICCV_2023_paper.pdf).

### env

```shell
conda create -n VQFont python=3.7
conda activate VQFont
cd VQ-Font
pip install -r requirements.txt
pip freeze > requirements.txt
```

### 训练字体来源

- https://github.com/aceliuchanghong/free-font

```shell
1.下载字体
2.字体转图片(gen_imgs_from_ttf.py)
3.字体json生成(to_hex.py)
4.图片文件夹划分
```

## Data Preparation

### Images and Characters

1) Collect a series of '.ttf'(TrueType) or '.otf'(OpenType) files to generate images for training models. and divide
   them into source content font and training set and test set. In order to better learn different styles, there should
   be differences and diversity in font styles in the training set.

2) Secondly, specify the characters to be generated (including training characters and test characters), eg the
   first-level Chinese character table contains 3500 Chinese characters.

> trian_val_3500: {乙、十、丁、厂、七、卜、人、入、儿、匕、...、etc}  
> train_3000: {天、成、在、麻、...、etc}  
> val_500: {熊、湖、战、...、etc}

3) Convert the characters in the second step into unicode encoding and save them in json format, you can convert the
   utf8 format to unicode by using ```hex(ord(ch))[2:].upper():```, examples can be found in ```./meta/```.

> trian_val_all_characters: ["4E00", "4E01", "9576", "501F", ...]  
> train_unis: ["4E00", "4E01", ...]  
> val_unis: ["9576", "501F", ...]

4) After that, draw all font images via ```./datasets/font2image.py```. All images are named
   by ```'characters + .png'```, such as ```‘阿.png’```.
   Organize directories structure as below, and ```train_3000.png``` means draw the image from
   train_unis: ["4E00", "4E01", ...].
   在vae的训练过程中,其实是训练emb模型,所以只需要content_font的字体图片即可
   其中vae_train.py需要分为3000:500的
   然后vae_emb.py需要全部的3500图片
   第二阶段few-shot的训练才需要需要上面收集的所有字体,转为图片,分为content+训练集和测试集即可

> Font Directory  
> |--| content  
> |&#8195; --| kaiti4train_VAE  
> |&#8195; &#8195; --| train_3000.png  
> |&#8195; &#8195; --| ...  
> |&#8195; --| kaiti4val_VAE  
> |&#8195; &#8195; --| val_500.png  
> |&#8195; &#8195; --| ...  
> |&#8195; --| kaiti4train_FFG  
> |&#8195; &#8195; --| trian_val_3500.png  
> |&#8195; &#8195; --| ...  
> |--| train  
> |&#8195; --| train_font1  
> |&#8195; --| train_font2  
> |&#8195; &#8195; --| trian_val_3500.png   
> |&#8195; &#8195; --| ...  
> |&#8195; --| ...  
> |--| val  
> |&#8195; --| val_font1  
> |&#8195; --| val_font2  
> |&#8195; &#8195; --| trian_val_3500.png    
> |&#8195; &#8195; --| ...  
> |&#8195; --| ...

### Build meta files and lmdb environment

1. 需要用`dataset/font2image.py`之类的先获取字体图片

2. 参照下面获取meta数据

Run script ```./build_trainset.sh``` or 查看 `./build_dataset/build_meta4train.py`里面的示例

 ```
  python3 ./build_dataset/build_meta4train.py \
  --saving_dir ./results/your_task_name/ \
  --content_font path\to\all_content \
  --train_font_dir path\to\training_font \
  --val_font_dir path\to\validation_font \
  --seen_unis_file path\to\train_unis.json \
  --unseen_unis_file path\to\val_unis.json 
  ```

## Training

The training process is divided into two stages:

1）Pre-training the content encoder and codebook
via [VQ-VAE](https://arxiv.org/abs/1711.00937)

`cd vae .. python vae_train.py` (在vae的训练过程中,其实是训练emb模型,所以只需要content_font的字体图片即可)

2）Training the few shot font generation model
via [GAN](https://dl.acm.org/doi/abs/10.1145/3422622).

`python3 train.py params....`

### Pre-train VQ-VAE

When pre-training VQ-VAE, the reconstructed character object comes from train_unis in the content font, The training
process can be found at ```./model/VQ-VAE.ipynb```.

Then use the pre-trained content encoder to calculate a similarity between all training and test characters and store it
as a dictionary.
> {'4E07': {'4E01': 0.2143, '4E03': 0.2374, ...}, '4E08': {'4E01': 0.1137, '4E03': 0.1020, ...}, ...}

### Few shot font generation

Modify the configuration in the file ```./cfgs/custom.yaml```

#### Keys

* work_dir: the root directory for saved results. (keep same with the `saving_dir` above)
* data_path: path to data lmdb environment. (`saving_dir/lmdb`)
* data_meta: path to train meta file. (`saving_dir/meta`)
* content_font: the name of font you want to use as source font.
* all_content_char_json: the json file which stores all train and val characters.
* other values are hyperparameters for training.

#### Run scripts

* ```
  python3 train.py task_name cfgs/custom.yaml
    #--resume \path\to\your\pretrain_model.pdparams
  ```

## Test

### Run scripts

* ```
  python3 inference.py ./cfgs/custom.yaml \
  --weight \path\to\saved_model.pdparams \
  --content_font \path\to\content_imgs \
  --img_path \path\to\test_imgs \
  --saving_root ./infer_res
  ```

## Explain

1. 图片含义:第一行参考字符 第二行GT 第三行模型生成字符

![0022000-comparable_ufuu_.png](z_using_files/training_log_pics/0022000-comparable_ufuu_.png)

2. 论文里面的FID、SSIM等指标是怎么计算的?

```
FID:：参考论文-Gans trained by a two time-scale update rule converge to a nash equilibrium. In NeurIPS, 2017.
LPIPS：参考论文-The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.
```

3. VQ-VAE时候,数据norm到[-0.5, 0.5],但是在阶段2, 喂给encoder的图片数据normalized [-1, 1].是否有问题?

```text
TODO:之后再看
```

4. 输出的png中的sfsu, sfuu, ufsu, ufuu分别是什么含义呢?

```text
SFSU等表示了不同字体和字符的组合情况（风格和字符都有训练集和测试集） (seen font seen unicode)
sfsu # 见过的字符见过的字体
sfuu # 没有见过的字符见过字体
ufsu # 见过的字符没见过的字体
ufuu_ # 没有见过的字符和字体
```

5. load_pretrain_vae_model函数似乎对于gen没有任何返回值,那么输入的gen就成为摆设了,是不是代码写得有问题?

```text
TODO:之后再看
```

```
vscode远程开发步骤
1.ssh <服务器登录名>@<公网ip>
2.输入服务器登录密码或者修改config 文件指定端口
3.往~/.ssh/authorized_keys里面添加C:\Users\【用户名】\.ssh”，访问“id_rsa.pub的数据(没有可以ssh-keygen)
```
