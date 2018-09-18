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
import argparse
import os

# 机器学习库
import tensorflow as tf

# 项目库
from LatentLearning.model import LatentLearning


# 参数设置
parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='迭代步数')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='批量大小')
parser.add_argument('--lr', dest='lr', type=float, default=0.00005, help='初始学习率')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoints/', help='models are saved here')
args = parser.parse_args()


def main(_):
    """
    LatentLearning main function
    :param _:
    :return:
    """
    # 创建训练/测试过程中所需的文件目录
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    # 创建会话/构建LatentLearning网络/训练网络/测试网络
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.80)
    with tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)) as sess:
        model = LatentLearning(sess, batch_size=args.batch_size, checkpoint_dir=args.checkpoint_dir)

        if args.phase == 'train':
            model.train(args)
        elif args.phase == 'test':
            model.test(args)
        elif args.phase == 'inference':
            model.inference(args)
        else:
            print("args.phase is train, test or inference!!")


if __name__ == '__main__':
    tf.app.run()