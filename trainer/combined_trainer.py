from .base_trainer import BaseTrainer
import utils
from datasets import cyclize
import torch

torch.autograd.set_detect_anomaly = True


class CombinedTrainer(BaseTrainer):
    """
    CombinedTrainer
    """

    def __init__(self, gen, disc, g_optim, d_optim, g_scheduler, d_scheduler,
                 logger, evaluator, cv_loaders, cfg):  # cls_char
        super().__init__(gen, disc, g_optim, d_optim, g_scheduler, d_scheduler,
                         logger, evaluator, cv_loaders, cfg)

    def train(self, loader, st_step=1, max_step=100000, component_embeddings=None, chars_sim_dict=None):
        # loader中存放了一个batch的数据
        """
        train
        """
        self.gen = self._get_model(self.gen)
        self.disc = self._get_model(self.disc)
        self.gen_ema = self._get_model(self.gen_ema)

        self.gen.train()
        if self.disc is not None:
            self.disc.train()

        # loss stats
        losses = utils.AverageMeters("g_total", "pixel", "disc", "gen", "contrastive")
        # discriminator stats
        discs = utils.AverageMeters("real_font", "real_uni", "fake_font", "fake_uni")
        # etc stats
        stats = utils.AverageMeters("B_style", "B_target")
        self.step = st_step
        self.clear_losses()
        self.logger.info("Start training FewShot ...")

        while True:
            for (in_style_ids, in_imgs, trg_style_ids, trg_uni_ids, trg_imgs,
                 content_imgs, trg_unis, style_sample_index, trg_sample_index, ref_unis) in cyclize(loader):
                """
                in_style_ids:reference font的index,长度为3
                in_imgs:reference image list
                trg_style_ids:生成的目标font的index,长度为1
                trg_uni_ids:生成目标字符的index
                trg_imgs:生成目标字符的GT image
                content_imgs:参考的内容字符image
                trg_unis:需要生成的字符、需要重构的字符
                style_sample_index:loader传入的index,长度为3
                trg_sample_index:目标的index
                len(loader)代表full train dataset需要迭代的次数
                """
                epoch = self.step // len(loader)
                B = trg_imgs.shape[0]
                stats.updates({
                    "B_style": in_imgs.shape[0],
                    "B_target": B
                })

                in_style_ids = in_style_ids.cuda()  # [font1 x 3,font2 x 3,...,fontn x 3];num=len(cfg.batch_size)
                in_imgs = in_imgs.cuda()  # [B*3*2,C,H,W]   [B*3*2,1,128,128] 每一个batch内有
                content_imgs = content_imgs.cuda()  # [B*2,C,H,W]
                trg_uni_disc_ids = trg_uni_ids.cuda()
                trg_style_ids = trg_style_ids.cuda()
                trg_imgs = trg_imgs.cuda()

                #  复制codebook为batch
                bs_component_embeddings = self.get_codebook_detach(component_embeddings)

                ####################################################
                # infer
                ####################################################

                # 得到风格特征
                self.gen.encode_write_comb(in_style_ids, style_sample_index, in_imgs[0])  # [B*3,256,16,16]

                # 生成目标图像
                out_1, style_components_1 = self.gen.read_decode(trg_style_ids, trg_sample_index,
                                                                 content_imgs[0],
                                                                 bs_component_embeddings,
                                                                 trg_unis,
                                                                 ref_unis,
                                                                 chars_sim_dict)  # fake_img && 变换后的特征 && qs风格化的部件

                self.gen.encode_write_comb(in_style_ids, style_sample_index, in_imgs[1])  # [B*3,256,16,16]

                _, style_components_2 = self.gen.read_decode(trg_style_ids, trg_sample_index,
                                                             content_imgs[1],
                                                             bs_component_embeddings,
                                                             trg_unis,
                                                             ref_unis,
                                                             chars_sim_dict)  # fake_img && 变换后的特征 && qs风格化的部件

                # reconstruct img
                self_infer_imgs, style_components, feat_recons = self.gen.infer(trg_style_ids, trg_imgs[0],
                                                                                trg_style_ids,
                                                                                trg_sample_index, content_imgs[0],
                                                                                bs_component_embeddings)

                real_font, real_uni = self.disc(trg_imgs[0], trg_style_ids,
                                                trg_uni_disc_ids[0::self.num_postive_samples])
                # GT图像以及font id和character id
                fake_font, fake_uni = self.disc(out_1.detach(), trg_style_ids,
                                                trg_uni_disc_ids[0::self.num_postive_samples])

                fake_font_recon, fake_uni_recon = self.disc(self_infer_imgs.detach(), trg_style_ids,
                                                            trg_uni_disc_ids[0::self.num_postive_samples])
                self.add_gan_d_loss(real_font, real_uni, fake_font + fake_font_recon,
                                    fake_uni + fake_uni_recon)

                # 辨别器计算梯度并更新参数(固定生成器的参数)
                self.d_backward()  # 计算反向传播求解梯度
                self.d_optim.step()  # 更新权重参数
                self.d_scheduler.step()  # 通过step_size来更新学习率
                self.d_optim.zero_grad()  # 清空梯度

                fake_font, fake_uni = self.disc(out_1, trg_style_ids, trg_uni_disc_ids[0::self.num_postive_samples])

                # reconstruction
                # fake_font_recon, fake_uni_recon = 0, 0
                fake_font_recon, fake_uni_recon = self.disc(self_infer_imgs, trg_style_ids,
                                                            trg_uni_disc_ids[0::self.num_postive_samples])
                self.add_gan_g_loss(real_font, real_uni, fake_font + fake_font_recon,
                                    fake_uni + fake_uni_recon)
                self.add_pixel_loss(out_1, trg_imgs[0], self_infer_imgs)
                self.style_contrastive_loss(style_components_1, style_components_2, self.batch_size)

                # 生成器参数反向传播并更新(固定辨别器的参数)
                self.g_backward()  # 计算反向传播求解梯度
                self.g_optim.step()  # 更新权重参数
                self.g_scheduler.step()  # 通过step_size来更新学习率
                self.g_optim.zero_grad()  # 清空梯度

                discs.updates({
                    "real_font": real_font.mean().item(),
                    "real_uni": real_uni.mean().item(),
                    "fake_font": fake_font.mean().item(),
                    "fake_uni": fake_uni.mean().item(),
                }, B)

                loss_dic = self.clear_losses()
                losses.updates(loss_dic, B)  # accum loss stats

                self.accum_g()
                if self.step % self.cfg['tb_freq'] == 0:
                    self.baseplot(losses, discs, stats)

                if self.step % self.cfg['print_freq'] == 0:
                    self.log(losses, discs, stats)
                    self.logger.debug("GPU Memory usage: max mem_alloc = %.1fM / %.1fM",
                                      torch.cuda.max_memory_allocated() / 1000 / 1000,
                                      torch.cuda.max_memory_reserved() / 1000 / 1000)
                    losses.resets()
                    discs.resets()
                    stats.resets()

                if self.step % self.cfg['val_freq'] == 0:
                    epoch = self.step / len(loader)
                    self.logger.info("Validation at Epoch = {:.3f}".format(epoch))
                    self.evaluator.cp_validation(self.gen_ema, self.cv_loaders, self.step, bs_component_embeddings,
                                                 chars_sim_dict)
                    self.save(loss_dic['g_total'], self.cfg['save'], self.cfg.get('save_freq', self.cfg['val_freq']))

                if self.step >= max_step:
                    break

                self.step += 1

            if self.step >= max_step:
                break

        self.logger.info("Iteration finished.")

    def log(self, losses, discs, stats):
        """
        ...
        Step      90: L1  0.5653   Contrastive 12.1395  D   3.564  G   1.065  B_stl   2.0  B_trg   2.0
        Step     100: L1  0.5561   Contrastive 11.5012  D   3.543  G   1.041  B_stl   2.0  B_trg   2.0
        Step     110: L1  0.5178   Contrastive 11.3973  D   3.532  G   1.077  B_stl   2.0  B_trg   2.0
        Step     120: L1  0.5150   Contrastive 11.2907  D   3.522  G   1.072  B_stl   2.0  B_trg   2.0
        Step     130: L1  0.5060   Contrastive 10.5871  D   3.525  G   1.088  B_stl   2.0  B_trg   2.0
        ...
        L1 loss 又称为“绝对误差损失”（Mean Absolute Error, MAE）值越小意味着模型生成的字体与目标字体在像素级别上越相似，模型的生成效果越好
        计算方式是预测值与真实值之间的绝对差值之和
        Contrastive：表示对比损失（Contrastive Loss）。这个值越小越好。对比损失越小，表示模型在区分不同类别样本与聚合相同类别样本方面做得越好
        在模型训练中，对比损失用于最大化不同类别样本之间的距离，最小化相同类别样本之间的距离，增强模型对不同类别特征的区分能力
        D：表示判别器损失（Discriminator Loss）。判别器损失一般也是越小越好。D值越小，表示判别器在区分真实与生成数据方面表现得越好
        在生成对抗网络（GAN）中，判别器的任务是区分真实数据和生成数据，D 反映了判别器在这方面的表现。
        G：表示生成器损失（Generator Loss）。生成器损失通常也是越小越好。G值越小，表示生成器能够更好地生成逼真的数据，让判别器更难以区分真假
        在GAN中，生成器的任务是生成看起来与真实数据相似的数据，G 反映了生成器在迷惑判别器方面的效果
        B_stl：表示风格差异（Style Bias）。 风格差异越小越好。B_stl越小，表示生成的字体与目标风格之间的差异越小，风格迁移效果越好
        它反映了生成的字体与目标风格之间的差异，通常是衡量生成结果与目标风格一致性的一个指标。
        B_trg：表示目标风格偏差（Target Bias）。目标风格偏差越小越好。B_trg值越小，表示生成的字体样式更接近目标风格，同时偏离源风格更少
        这个值可能反映了生成的字体样式在逼近目标风格时与源字体风格之间的偏离程度。
        """
        self.logger.info(
            "  Step {step:7d}: L1 {L.pixel.avg:7.4f}   Contrastive {L.contrastive.avg:7.4f}"
            "  D {L.disc.avg:7.3f}  G {L.gen.avg:7.3f}"
            "  B_stl {S.B_style.avg:5.1f}  B_trg {S.B_target.avg:5.1f}"
            .format(step=self.step, L=losses, D=discs, S=stats))
