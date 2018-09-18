# -*- coding:utf-8 -*-

"""
@Brief
main function：parameter setting

@Description
Latent Learning idea verification

@Author: Ling Bao
@Data: September 18, 2018
@Version: 0.1.0
"""

# 系统库
from __future__ import division
import os
import time
from glob import glob
from six.moves import xrange
from matplotlib import pyplot as plt_model

# 项目库
from LatentLearning.lib_ops.ops import *

# 机器学习库
import tensorflow as tf

slim = tf.contrib.slim


class LatentLearning(object):
    def __init__(self, sess, batch_size=16, checkpoint_dir=None):
        # 通用变量
        self.sess = sess
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir

        # 6 构建模型
        self.build_model()

    def build_model(self):
        # ××××××××××××××××××××××××××××××××××××××××××××××××前期准备××××××××××××××××××××××××××××××××××××××××××××××××××××× #
        # 0 前期准备
        # 0.1 small判别器与生成器输入尺寸
        w_small = int(self.image_size / 2)
        h_small = int(self.image_size / 2)

        # 0.2 large模型输入数据
        c_ = self.input_c_dim + self.output_c_dim
        self.real_data = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, c_],
                                        name="image_and_mp")
        self.real_im = self.real_data[:, :, :, :self.input_c_dim]
        self.real_mp = self.real_data[:, :, :, self.input_c_dim:c_]

        # 0.3 small模型输入数据
        small_im_1 = self.real_im[:, :w_small, :h_small, :]
        small_im_2 = self.real_im[:, w_small:w_small + h_small, :h_small, :]
        small_im_3 = self.real_im[:, :w_small, h_small:h_small + w_small, :]
        small_im_4 = self.real_im[:, w_small:w_small + h_small, h_small:h_small + w_small, :]
        small_concat_im = tf.concat([small_im_1, small_im_2, small_im_3, small_im_4], 0)

        small_mp_1 = self.real_mp[:, :w_small, :h_small, :]
        small_mp_2 = self.real_mp[:, w_small:w_small + h_small, :h_small, :]
        small_mp_3 = self.real_mp[:, :w_small, h_small:h_small + w_small, :]
        small_mp_4 = self.real_mp[:, w_small:w_small + h_small, h_small:h_small + w_small, :]
        small_concat_mp = tf.concat([small_mp_1, small_mp_2, small_mp_3, small_mp_4], 0)

        # 0.4 VGG2网络初始化，用于感知损失计算
        vgg2 = VGG2()
        vgg2.vgg_2_load()

        # ××××××××××××××××××××××××××××××××××××××××××××××small部分××××××××××××××××××××××××××××××××××××××××××××××××××××× #
        # 1 small部分
        # 1.1 D_small损失
        # 1.1.1 获取输入数据
        self.real_im_small = small_concat_im
        self.real_mp_small = small_concat_mp
        self.fake_mp_small = self.generator_small(self.real_im_small, 4 * self.batch_size)

        # 1.1.2 真假判别
        self.real_concat_small = tf.concat([self.real_im_small, self.real_mp_small], 3)
        self.fake_concat_small = tf.concat([self.real_im_small, self.fake_mp_small], 3)
        self.d_s_x, self.d_s_y = self.discriminator_small(self.real_concat_small, 4 * self.batch_size, reuse=False)
        self.d_s_x_, self.d_s_y_ = self.discriminator_small(self.fake_concat_small, 4 * self.batch_size, reuse=True)

        # 1.1.3 small判别器对抗损失
        self.d_s_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_s_x, labels=tf.ones_like(self.d_s_y)))
        self.d_s_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_s_x_, labels=tf.zeros_like(self.d_s_y_)))
        self.d_small_loss_a = self.d_s_loss_real + self.d_s_loss_fake

        # 1.1.4 small判别器训练概要设置(**后续再考虑，用于追踪损失与生成器图像质量**)
        self.d_s_real_sum = tf.summary.histogram("d_small_real", self.d_s_y)
        self.d_s_fake_sum = tf.summary.histogram("d_small_fake", self.d_s_y_)
        self.g_s_fake_sum = tf.summary.image("g_small", self.fake_mp_small)
        self.d_s_loss_sum = tf.summary.scalar("d_s_loss", self.d_small_loss_a)

        # 1.2 G_small损失
        # 1.2.1 small生成器对抗损失
        self.g_small_loss_a = self.d_s_loss_fake

        # 1.2.2 L2损失--Euclidean loss
        self.g_small_loss_e = tf.reduce_mean(
            tf.abs(self.real_mp_small - self.fake_mp_small) * tf.abs(self.real_mp_small - self.fake_mp_small))

        # 1.2.3 small生成器感知损失
        vgg2.x = self.real_mp_small
        vgg2.vgg_2()
        f_real_mp_small = vgg2.net
        vgg2.x = self.fake_mp_small
        vgg2.vgg_2()
        f_fake_mp_small = vgg2.net
        self.g_small_loss_p = tf.reduce_mean(tf.abs(f_real_mp_small - f_fake_mp_small)
                                             * tf.abs(f_real_mp_small - f_fake_mp_small))

        # 1.2.4 small生成器第一部分损失
        self.g_small_loss_one = \
            self.g_small_loss_a + self.lambda_E * self.g_small_loss_e + self.lambda_P * self.g_small_loss_p

        # 1.2.5 small生成器训练概要设置
        self.g_s_loss_a_sum = tf.summary.scalar("g_s_loss_a", self.g_small_loss_a)
        self.g_s_loss_e_sum = tf.summary.scalar("g_s_loss_e", self.g_small_loss_e)
        self.g_s_loss_p_sum = tf.summary.scalar("g_s_loss_p", self.g_small_loss_p)
        self.g_s_loss_one_sum = tf.summary.scalar("g_s_loss_one", self.g_small_loss_one)

        # ××××××××××××××××××××××××××××××××××××××××××××××large部分××××××××××××××××××××××××××××××××××××××××××××××××××××× #
        # 2 large部分
        # 2.1 D_large损失
        # 2.1.1 获取输入数据
        self.real_im_large = self.real_im
        self.real_mp_large = self.real_mp
        self.fake_mp_large = self.generator_large(self.real_im_large, self.batch_size)

        # 2.1.2 真假判别
        self.real_concat_large = tf.concat([self.real_im_large, self.real_mp_large], 3)
        self.fake_concat_large = tf.concat([self.real_im_large, self.fake_mp_large], 3)
        self.d_l_x, self.d_l_y = self.discriminator_large(self.real_concat_large, self.batch_size, reuse=False)
        self.d_l_x_, self.d_l_y_ = self.discriminator_large(self.fake_concat_large, self.batch_size, reuse=True)

        # 2.1.3 large判别器对抗损失
        self.d_l_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_l_x, labels=tf.ones_like(self.d_l_y)))
        self.d_l_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_l_x_, labels=tf.zeros_like(self.d_l_y_)))
        self.d_large_loss_a = self.d_l_loss_real + self.d_l_loss_fake

        # 2.1.4 large判别器训练概要设置(**后续再考虑，用于追踪损失与生成器图像质量**)
        self.d_l_real_sum = tf.summary.histogram("d_large_real", self.d_l_y)
        self.d_l_fake_sum = tf.summary.histogram("d_large_fake", self.d_l_y_)
        self.g_l_fake_sum = tf.summary.image("g_large", self.fake_mp_large)
        self.d_l_loss_sum = tf.summary.scalar("d_l_loss", self.d_large_loss_a)

        # 2.2 G_large损失
        # 2.2.1 large生成器对抗损失
        self.g_large_loss_a = self.d_l_loss_fake

        # 2.2.2 L2损失--Euclidean loss
        self.g_large_loss_e = tf.reduce_mean(
            tf.abs(self.real_mp_large - self.fake_mp_large) * tf.abs(self.real_mp_large - self.fake_mp_large))

        # 2.2.3 large生成器感知损失
        vgg2.x = self.real_mp_large
        vgg2.vgg_2()
        f_real_mp_large = vgg2.net
        vgg2.x = self.fake_mp_large
        vgg2.vgg_2()
        f_fake_mp_large = vgg2.net
        self.g_large_loss_p = tf.reduce_mean(tf.abs(f_real_mp_large - f_fake_mp_large)
                                             * tf.abs(f_real_mp_large - f_fake_mp_large))

        # 2.2.4 large生成器第一部分损失
        self.g_large_loss_one = \
            self.g_large_loss_a + self.lambda_E * self.g_large_loss_e + self.lambda_P * self.g_large_loss_p

        # 2.2.5 large生成器训练概要设置
        self.g_l_loss_a_sum = tf.summary.scalar("g_l_loss_a", self.g_large_loss_a)
        self.g_l_loss_e_sum = tf.summary.scalar("g_l_loss_e", self.g_large_loss_e)
        self.g_l_loss_p_sum = tf.summary.scalar("g_l_loss_p", self.g_large_loss_p)
        self.g_l_loss_one_sum = tf.summary.scalar("g_sl_loss_one", self.g_large_loss_one)

        # ×××××××××××××××××××××××××××××××××××××××××××××交叉尺度损失××××××××××××××××××××××××××××××××××××××××××××××××××××× #
        # 3 交叉尺度损失
        # 3.1 获取large与small判别器生成图片
        self.fake_mp_small_ = self.fake_mp_small
        fml = self.fake_mp_large
        fml_1 = fml[:, :w_small, :h_small, :]
        fml_2 = fml[:, w_small:w_small + h_small, :h_small, :]
        fml_3 = fml[:, :w_small, h_small:h_small + w_small, :]
        fml_4 = fml[:, w_small:w_small + h_small, h_small:h_small + w_small, :]
        self.fake_mp_large_ = tf.concat([fml_1, fml_2, fml_3, fml_4], 0)

        # 3.2 计算交叉尺度损失
        cc_loss = tf.reduce_mean(
            tf.abs(self.fake_mp_small_ - self.fake_mp_large_) * tf.abs(self.fake_mp_small_ - self.fake_mp_large_))
        self.cross_scale_loss_two = self.lambda_C * cc_loss

        # 3.3 交叉尺度损失训练概要设置
        self.cc_loss_sum = tf.summary.scalar("cross_scale_loss", self.cross_scale_loss_two)

        # ××××××××××××××××××××××××××××××××××××××××××××××生成器总损失×××××××××××××××××××××××××××××××××××××××××××××××××××× #
        # 4 生成器总损失
        # 4.1 small生成器总损失
        self.g_s_loss = self.g_small_loss_one + self.cross_scale_loss_two

        # 4.2 large生成器总损失
        self.g_l_loss = self.g_large_loss_one + self.cross_scale_loss_two

        # 4.3 生成器总损失训练概要设置
        self.g_s_loss_sum = tf.summary.scalar("g_s_loss", self.g_s_loss)
        self.g_l_loss_sum = tf.summary.scalar("g_l_loss", self.g_l_loss)

        # 5 模型参数训练与保存
        t_vars = tf.trainable_variables()
        self.d_l_vars = [var for var in t_vars if 'd_L_' in var.name]
        self.g_l_vars = [var for var in t_vars if 'g_L_' in var.name]
        self.d_s_vars = [var for var in t_vars if 'd_S_' in var.name]
        self.g_s_vars = [var for var in t_vars if 'g_S_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, args):
        # 设置优化器
        d_s_op = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.d_small_loss_a, var_list=self.d_s_vars)
        d_l_op = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.d_large_loss_a, var_list=self.d_l_vars)
        g_s_op = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.g_s_loss, var_list=self.g_s_vars)
        g_l_op = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.g_l_loss, var_list=self.g_l_vars)

        # 初始化变量并创建会话
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # 合并概要并写图结构到日志文件
        self.merged_summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        # 如果存在已保存模型断点，则进行模型载入
        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        counter = 1
        for epoch in xrange(args.epoch):
            # 获取训练数据路径列表
            data = glob('{}/*.jpg'.format(args.train_im_dir))
            np.random.shuffle(data)

            # 配置最大训练样本数目
            batch_idx_set = min(len(data), args.train_size)
            batch_idx_set /= self.batch_size
            batch_idx_set = int(np.floor(batch_idx_set))

            # 开始进行本批次样本训练
            for idx in xrange(0, batch_idx_set):
                # 获取本轮训练的数据
                batch_files = data[idx * self.batch_size: (idx + 1) * self.batch_size]
                batch = [load_data(batch_file, args) for batch_file in batch_files]

                # 转换为numpy数组
                batch_images = np.array(batch).astype(np.float32)

                # 更新large判别器网络/large生成器网络/small判别器网络/small生成器网络
                _ = self.sess.run([d_l_op], feed_dict={self.real_data: batch_images})
                _ = self.sess.run([g_l_op], feed_dict={self.real_data: batch_images})
                _ = self.sess.run([d_s_op], feed_dict={self.real_data: batch_images})
                _ = self.sess.run([g_s_op], feed_dict={self.real_data: batch_images})

                # 记录全局迭代步数
                counter += 1

                # 保存概述数据
                if np.mod(counter, 100) == 0:
                    summary_str = self.sess.run(self.merged_summary_op, feed_dict={self.real_data: batch_images})
                    self.writer.add_summary(summary_str, counter)

                    f_l = self.fake_mp_large.eval({self.real_data: batch_images})
                    f_s = self.fake_mp_small.eval({self.real_data: batch_images})

                    r_sum = sum(sum(batch[0][:, :, 3]))
                    f_l_sum = sum(sum(sum(f_l[0]))) / 3
                    f_s_sum = sum(sum(sum(f_s[0]))) / 3
                    print("\n******************************************************************")
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, real: %.4f, l_fake: %.4f, s_fake: %.4f"
                          % (epoch, idx, batch_idx_set, time.time() - start_time, r_sum, f_l_sum, f_s_sum))
                    print("******************************************************************\n")

                    im_path = "./sample/"
                    im_name = "fake_large_" + str(epoch) + ".jpg"
                    cv2.imwrite(im_path + im_name, f_l[0])

                # 打印每一步训练过程信息
                if np.mod(counter, 10) == 0:
                    # 获取损失模型损失
                    err_d_s_a = self.d_small_loss_a.eval({self.real_data: batch_images})
                    err_d_l_a = self.d_large_loss_a.eval({self.real_data: batch_images})
                    err_g_s = self.g_s_loss.eval({self.real_data: batch_images})
                    err_g_l = self.g_l_loss.eval({self.real_data: batch_images})

                    # 打印训练信息
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_s_a_loss: %.8f, d_l_a_loss: %.8f, g_s_loss: %.8f,"
                          " g_l_loss: %.8f" % (epoch, idx, batch_idx_set, time.time() - start_time, err_d_s_a,
                                               err_d_l_a, err_g_s, err_g_l))

                # # 每训练固定批次便进行一次验证（此次为200批次）
                # if np.mod(counter, 400) == 0:
                #     self.sample_model(args)

                # 每训练固定批次便进行一次模型保存（此次为500批次）
                if np.mod(counter, 5000) == 0:
                    self.save(args.checkpoint_dir, counter)

    def test(self, args):
        # 如果存在已保存模型断点，则进行模型载入
        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # 获取训练数据路径列表
        data = glob('{}/*.jpg'.format(args.test_im_dir))

        # 配置最大训练样本数目
        batch_idx_set = len(data)
        batch_idx_set /= self.batch_size
        batch_idx_set = int(np.floor(batch_idx_set))

        # 计算平均绝对误差与平均均方误差
        sum_mae = 0.0
        sum_mse = 0.0

        # 开始进行本批次样本训练
        for idx in xrange(0, batch_idx_set):
            # 获取本轮训练的数据
            batch_files = data[idx * self.batch_size: (idx + 1) * self.batch_size]
            batch = [load_data(batch_file, args) for batch_file in batch_files]

            # 转换为numpy数组
            batch_images = np.array(batch).astype(np.float32)

            f_l = self.fake_mp_large.eval({self.real_data: batch_images})

            r_sum = sum(sum(batch[0][:, :, 3]))
            f_l_sum = sum(sum(sum(f_l[0]))) / 3
            abs_tmp = abs(r_sum - f_l_sum)
            sqr_tmp = pow(r_sum - f_l_sum, 2)

            print("Image: [%4d/%4d] time: %4.4f, real: %.4f, l_fake: %.4f, abs_diff: %.4f, sqr_diff: %.4f"
                  % (idx, batch_idx_set, time.time() - start_time, r_sum, f_l_sum, abs_tmp, sqr_tmp))

            sum_mae += abs_tmp
            sum_mse += sqr_tmp

            mp = np.mean(f_l[0], axis=2)
            mp_name = batch_files[0].split("/")[-1].split('.')[0]
            plt_model.imsave(args.test_dir + mp_name + ".png", mp, cmap=plt_model.get_cmap('jet'))
            cv2.imwrite(args.test_dir + mp_name + ".jpg", batch[0][:, :, :3])

        mae = sum_mae / batch_idx_set
        mse = np.sqrt(sum_mse / batch_idx_set)
        print("\n******************************************************************")
        print("MAE: %.8f, MSE: %.8f" % (mae, mse))
        print("******************************************************************\n")

    def inference(self, img, mp_name):
        """
        用于人群密度估计推理
        :param img: 待估计图片
        :param mp_name: 密度图名称
        :return: None
        """
        # 如果存在已保存模型断点，则进行模型载入
        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        mp_tmp = self.sess.run([self.fake_mp_large], feed_dict={self.real_data: img_tmp_np})
        run_time = time.time() - start_time

        return mp_tmp, run_time

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "models"
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def generator_large(self, image, batch_size, reuse=False):
        """
        Large生成器网络
        :param image: 输入数据
        :param batch_size
        :param reuse:
        :return: 生成图片
        """
        with tf.variable_scope("generator_large"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

        h, w = self.h, self.w
        f1 = lrelu(linear(tf.reshape(image, [batch_size, -1]), h, 'm_cls_h1_lin'))
        f2 = lrelu(linear(tf.reshape(f1, [batch_size, -1]), 2 * h, 'm_cls_h2_lin'))
        f3 = lrelu(linear(tf.reshape(f2, [batch_size, -1]), h, 'm_cls_h3_lin'))
        f4 = linear(tf.reshape(f3, [batch_size, -1]), 1, 'm_cls_h4_lin')
        f4_ = tf.reshape(f4, [-1])

        pre_y = tf.nn.relu(tf.nn.sigmoid(f4_))

        return pre_y

    def save(self, checkpoint_dir, step):
        model_name = "latent.model"
        model_dir = "models"
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
