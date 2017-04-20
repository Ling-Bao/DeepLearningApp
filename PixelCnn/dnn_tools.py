# -*- coding: utf-8 -*-

# 系统模块
import re
import tensorflow as tf
import cv2
import os

tensor_board_name = 'tbn'


# ************************************************************************************* #
# 一、TensorFlow打包数据
#
# (1) 图片与类别标签
# (2) 图片与xml
# (3) 图片与密度图
#
# ************************************************************************************* #

def encode_to_tf_records(label_file, data_root, new_name='data.tf_records', resize=None):
    """把train.txt文件格式,每一行:图片路径名   类别标签; 将数据打包,转换成tf_records格式,以便后续高效读取

    Args:
        label_file: 标签文件--*.txt文件格式,每一行:图片路径名   类别标签
        data_root: 图像文件根目录
        new_name:
        resize: 将图像进行resize;若为None则不进行

    Returns:
        无
    """
    writer = tf.python_io.TFRecordWriter(data_root + '/' + new_name)

    num_example = 0
    with open(label_file, 'r') as f:
        for l in f.readlines():
            l = l.split()
            image = cv2.imread(data_root + "/" + l[0])
            if resize is not None:
                image = cv2.resize(image, resize)
            height, width, n_channel = image.shape

            label = int(l[1])

            example = tf.train.Example(features=tf.train.Features(feature={
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                'n_channel': tf.train.Feature(int64_list=tf.train.Int64List(value=[n_channel])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))

            serialized = example.SerializeToString()
            writer.write(serialized)
            num_example += 1

    print label_file, "样本数据量：", num_example

    writer.close()


def decode_from_tf_records(filename, num_epoch=None):
    """读取tf_records文件

    Args:
        filename: tf_records格式文件名列表
        num_epoch: 每步数量

    Returns:
        无
    """
    # 因为有的训练数据过于庞大，被分成了很多个文件，所以第一个参数就是文件列表名参数
    filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epoch)
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)

    example = tf.parse_single_example(serialized, features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'n_channel': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })

    label = tf.cast(example['label'], tf.int32)
    image = tf.decode_raw(example['image'], tf.uint8)
    image = tf.reshape(image, tf.stack([
        tf.cast(example['height'], tf.int32),
        tf.cast(example['width'], tf.int32),
        tf.cast(example['n_channel'], tf.int32)]))

    return image, label


def get_batch(image, label, batch_size, crop_size=80):
    """根据队列流数据格式，解压出一张图片后，输入一张图片，对其做预处理及样本随机扩充

    Args:
        image: 图像数据
        label: 标签
        batch_size: 批量大小
        crop_size: 裁剪大小

    Returns:
        无
    """
    # 数据扩充变换
    distorted_image = tf.random_crop(image, [crop_size, crop_size, 3])  # 随机裁剪
    distorted_image = tf.image.random_flip_up_down(distorted_image)  # 上下随机翻转
    # distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)  # 亮度变化
    # distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)  # 对比度变化

    # 生成batch
    # shuffle_batch的参数：
    # capacity用于定义shuffle的范围，如果是对整个训练数据集，获取batch，那么capacity就应该够大以保证数据足够乱
    images, label_batch = tf.train.shuffle_batch([distorted_image, label], batch_size=batch_size,
                                                 num_threads=16, capacity=50000, min_after_dequeue=10000)
    # images, label_batch = tf.train.batch([distorted_image, label], batch_size=batch_size)

    # 调试显示
    # tf.image_summary('images', images)
    return images, tf.reshape(label_batch, [batch_size])


def get_test_batch(image, label, batch_size, crop_size):
    """用于测试阶段，使用的get_batch函数

        Args:
            image: 图像数据
            label: 标签
            batch_size: 批量大小
            crop_size: 裁剪大小

        Returns:
            无
    """
    # 数据扩充变换
    distorted_image = tf.image.central_crop(image, 39. / 45.)
    distorted_image = tf.random_crop(distorted_image, [crop_size, crop_size, 3])  # 随机裁剪

    images, label_batch = tf.train.batch([distorted_image, label], batch_size=batch_size)

    return images, tf.reshape(label_batch, [batch_size])


def test_tfrecord():
    """测试上面的压缩、解压代码

        Args:
            image: 图像数据
            label: 标签
            batch_size: 批量大小
            crop_size: 裁剪大小

        Returns:
            无
    """
    encode_to_tf_records("data/train.txt", "data", resize=(100, 100))
    image, label = decode_from_tf_records('data/data.tf_records')
    batch_image, batch_label = get_batch(image, label, 3)  # batch 生成测试

    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for l in range(100):  # 每run一次，就会指向下一个样本，一直循环
            # image_np,label_np=session.run([image,label]) #每调用run一次，那么
            # '''cv2.imshow("temp",image_np)
            # cv2.waitKey()'''
            # print label_np
            # print image_np.shape

            batch_image_np, batch_label_np = session.run([batch_image, batch_label])

            print batch_image_np.shape
            print batch_label_np.shape

        coord.request_stop()  # queue需要关闭，否则报错
        coord.join(threads)


# ************************************************************************************* #
# 二、网络架构与训练
#
# (1) 网络结构
# (2) 网络优化方法
# (3) 训练，数据输入网络
#
# ************************************************************************************* #


class Network(object):
    def __init__(self):
        with tf.variable_scope("weights"):
            self.weights = {
                # 39*39*3 --> 36*36*20 --> 18*18*20
                'conv1': tf.get_variable('conv1', [4, 4, 3, 20],
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                # 18*18*20 --> 16*16*40 --> 8*8*40
                'conv2': tf.get_variable('conv2', [3, 3, 20, 40],
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                # 8*8*40 --> 6*6*60 --> 3*3*60
                'conv3': tf.get_variable('conv3', [3, 3, 40, 60],
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                # 3*3*60 --> 120
                'fc1': tf.get_variable('fc1', [3 * 3 * 60, 120], initializer=tf.contrib.layers.xavier_initializer()),
                # 120 --> 6
                'fc2': tf.get_variable('fc2', [120, 6], initializer=tf.contrib.layers.xavier_initializer()),
            }

        with tf.variable_scope("biases"):
            self.biases = {
                'conv1': tf.get_variable('conv1', [20, ],
                                         initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv2': tf.get_variable('conv2', [40, ],
                                         initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv3': tf.get_variable('conv3', [60, ],
                                         initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc1': tf.get_variable('fc1', [120, ],
                                       initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc2': tf.get_variable('fc2', [6, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))

            }

        with tf.variable_scope("losses"):
            self.cost = 0

    def inference(self, images):
        # 向量转为矩阵
        images = tf.reshape(images, shape=[-1, 39, 39, 3])  # [batch, in_height, in_width, in_channels]
        images = (tf.cast(images, tf.float32) / 255. - 0.5) * 2  # 归一化处理

        # 第一层
        conv1 = tf.nn.bias_add(tf.nn.conv2d(images, self.weights['conv1'], strides=[1, 1, 1, 1], padding='VALID'),
                               self.biases['conv1'])

        relu1 = tf.nn.relu(conv1)
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # 第二层
        conv2 = tf.nn.bias_add(tf.nn.conv2d(pool1, self.weights['conv2'], strides=[1, 1, 1, 1], padding='VALID'),
                               self.biases['conv2'])
        relu2 = tf.nn.relu(conv2)
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # 第三层
        conv3 = tf.nn.bias_add(tf.nn.conv2d(pool2, self.weights['conv3'], strides=[1, 1, 1, 1], padding='VALID'),
                               self.biases['conv3'])
        relu3 = tf.nn.relu(conv3)
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # dropout层，先把特征图转为向量
        flatten = tf.reshape(pool3, [-1, self.weights['fc1'].get_shape().as_list()[0]])
        drop1 = tf.nn.dropout(flatten, 0.5)

        # 全连接层1
        fc1 = tf.matmul(drop1, self.weights['fc1']) + self.biases['fc1']
        fc_relu1 = tf.nn.relu(fc1)

        # 全连接层2
        fc2 = tf.matmul(fc_relu1, self.weights['fc2']) + self.biases['fc2']

        return fc2

    def inference_test(self, images):
        # 向量转为矩阵
        images = tf.reshape(images, shape=[-1, 39, 39, 3])  # [batch, in_height, in_width, in_channels]
        images = (tf.cast(images, tf.float32) / 255. - 0.5) * 2  # 归一化处理

        # 第一层
        conv1 = tf.nn.bias_add(tf.nn.conv2d(images, self.weights['conv1'], strides=[1, 1, 1, 1], padding='VALID'),
                               self.biases['conv1'])

        relu1 = tf.nn.relu(conv1)
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # 第二层
        conv2 = tf.nn.bias_add(tf.nn.conv2d(pool1, self.weights['conv2'], strides=[1, 1, 1, 1], padding='VALID'),
                               self.biases['conv2'])
        relu2 = tf.nn.relu(conv2)
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # 第三层
        conv3 = tf.nn.bias_add(tf.nn.conv2d(pool2, self.weights['conv3'], strides=[1, 1, 1, 1], padding='VALID'),
                               self.biases['conv3'])
        relu3 = tf.nn.relu(conv3)
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # 全连接层1，先把特征图转为向量
        flatten = tf.reshape(pool3, [-1, self.weights['fc1'].get_shape().as_list()[0]])

        fc1 = tf.matmul(flatten, self.weights['fc1']) + self.biases['fc1']
        fc_relu1 = tf.nn.relu(fc1)

        fc2 = tf.matmul(fc_relu1, self.weights['fc2']) + self.biases['fc2']

        return fc2

    # 计算softmax交叉熵损失函数
    def softmax_loss(self, predicts, labels):
        predicts = tf.nn.softmax(predicts)
        labels = tf.one_hot(labels, self.weights['fc2'].get_shape().as_list()[1])
        loss = -tf.reduce_mean(labels * tf.log(predicts))  # tf.nn.softmax_cross_entropy_with_logits(predicts, labels)
        self.cost = loss

        return self.cost

    # 梯度下降
    def optimizer(self, loss, lr=0.001):
        train_optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

        return train_optimizer


def train():
    encode_to_tf_records("data/train.txt", "data", 'train.tfrecords', (45, 45))
    image, label = decode_from_tf_records('data/train.tfrecords')
    batch_image, batch_label = get_batch(image, label, batch_size=50, crop_size=39)  # batch 生成测试

    # 网络链接,训练所用
    net = Network()
    inf = net.inference(batch_image)
    loss = net.softmax_loss(inf, batch_label)
    optimizer = net.optimizer(loss)

    # 验证集所用
    encode_to_tf_records("data/val.txt", "data", 'val.tfrecords', (45, 45))
    test_image, test_label = decode_from_tf_records('data/val.tfrecords', num_epoch=None)
    test_images, test_labels = get_test_batch(test_image, test_label, batch_size=120, crop_size=39)  # batch 生成测试
    test_inf = net.inference_test(test_images)
    correct_prediction = tf.equal(tf.cast(tf.argmax(test_inf, 1), tf.int32), test_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        max_iter = 100000
        iter = 0
        if os.path.exists(os.path.join("model", 'model.ckpt')) is True:
            tf.train.Saver(max_to_keep=None).restore(session, os.path.join("model", 'model.ckpt'))
        while iter < max_iter:
            loss_np, _, label_np, image_np, inf_np = session.run([loss, optimizer, batch_label, batch_image, inf])
            # print image_np.shape
            # cv2.imshow(str(label_np[0]),image_np[0])
            # print label_np[0]
            # cv2.waitKey()
            # print label_np
            if iter % 50 == 0:
                print 'trainloss:', loss_np
            if iter % 500 == 0:
                accuracy_np = session.run([accuracy])
                print '***************test accruacy:', accuracy_np, '*******************'
                tf.train.Saver(max_to_keep=None).save(session, os.path.join('model', 'model.ckpt'))
            iter += 1

        coord.request_stop()  # queue需要关闭，否则报错
        coord.join(threads)


# ************************************************************************************* #
# 三、用于可视化跟踪神经网络模型中的变量
#
# (1) 在源码中加入需要跟踪的变量
# (2) 定义执行操作
# (3) 在session中定义保存路径
# (4) 在session执行的时保存
# (5) 训练完毕后，直接在终端输入命令并打开浏览器
#
# ************************************************************************************* #

def activation_summary(x):
    """用于帮助创建TensorBoard Summaries,提供直方图和稀疏图

    Args:
        x: Tensor变量

    Returns:
        无
    """
    tensor_name = re.sub('%s/' % tensor_board_name, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def add_loss_summaries(total_loss):
    """添加误差率summaries,为所有损失和相关摘要生成移动平均值,并可视化网络的性能

    Args:
        total_loss: 总损失loss()

    Returns:
        损失的移动平均值
    """
    # 计算单个损失及总损失的移动平均值
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # 增加单个损失及总损失的scalar摘要; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


if __name__ == '__main__':
    # tfrecord test
    test_tfrecord()

    # A simple class of Network for Classify picture
    train()