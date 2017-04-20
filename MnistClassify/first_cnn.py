# -*- coding: utf-8 -*-
"""
@Author: Ling Bao
@Date: September 30, 2016

@Function：Tensorflow 手写数字识别

学习
"""
# 系统库
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 设置参数
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, '是否使用假数据进行单元测试')
flags.DEFINE_integer('max_steps', 5000, '训练最大步数')
flags.DEFINE_float('learning_rate', 0.001, '初始学习率')
flags.DEFINE_float('dropout', 0.5, 'dropout概率')
flags.DEFINE_string('data_dir', './MNIST_data', '训练数据所在目录')
flags.DEFINE_string('summaries_dir', './log', '摘要存放路径')


def train():
    # 获取训练数据
    mnist = input_data.read_data_sets(FLAGS.data_dir,
                                      one_hot=True,
                                      fake_data=FLAGS.fake_data)

    # 创建会话
    sess = tf.InteractiveSession()

    with tf.name_scope('input'):
        """图片与标签变量"""
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

    with tf.name_scope('input_reshape'):
        """对输入x进行reshape, 变为28*28矩阵"""
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.image_summary('input', image_shaped_input, 10)

    def weight_variable(shape):
        """对权重进行合理的初始化"""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        """对偏差进行合理的初始化"""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        """卷积函数"""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        """池化函数"""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def variable_summaries(var, name):
        """将Summaries添加到一个Tensor"""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.scalar_summary('stddev/' + name, stddev)
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))
            tf.histogram_summary(name, var)

    def nn_layer(input_tensor, con_width, con_heigh, input_dim, output_dim, layer_name, act=tf.nn.relu):
        """
        可重复使用的简单神经网络层

        包括：w*x+b, Conv, ReLU, MaxPooling
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                """设置并保存权重的状态"""
                weights = weight_variable([con_width, con_heigh,  input_dim, output_dim])
                variable_summaries(weights, layer_name + '/weights')
        
            with tf.name_scope('biases'):
                """设置并保存偏差的状态"""
                biases = bias_variable([output_dim])
                variable_summaries(biases, layer_name + '/biases')

            with tf.name_scope('Wx_plus_b'):
                """计算卷积"""
                preactivate = conv2d(input_tensor, weights) + biases
                tf.histogram_summary(layer_name + '/pre_activations', preactivate)

            # ReLU
            activations = act(preactivate, name='activation')
            tf.histogram_summary(layer_name + '/activations', activations)

            # 最大池化
            max_pooling = max_pool_2x2(activations)
            tf.histogram_summary(layer_name + '/maxpooling', max_pooling)
        
            return max_pooling

    # 构建两层卷积神经网络Conv+ReLU+Maxpooling
    # 1*28*28 --> 32*28*28 --> 32*14*14
    hidden_pool1 = nn_layer(image_shaped_input, 5, 5, 1, 32, 'layer_1')

    # 32*14*14 --> 64*14*14 --> 64*7*7
    hidden_pool2 = nn_layer(hidden_pool1, 5, 5, 32, 64, 'layer_2')
    
    # 密集链接层Matmul + ReLU
    # 64*7*7 --> 3136 --> 1024
    with tf.name_scope('layer_fc1'):
        """初始化权重与偏值"""
        layer_name = 'layer_fc1'
        w_fc1 = weight_variable([7 * 7 * 64, 1024])
        variable_summaries(w_fc1, layer_name + '/weights')
        b_fc1 = bias_variable([1024])
        variable_summaries(b_fc1, layer_name + '/biases')

        # Reshape + Matmul + ReLU
        hidden_pool2_flat = tf.reshape(hidden_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(hidden_pool2_flat, w_fc1) + b_fc1)

    # 1024 --> 1024
    with tf.name_scope('dropout'):
        """Dropout层"""
        keep_prob = tf.placeholder(tf.float32)
        tf.scalar_summary('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(h_fc1, keep_prob)

    # 1024 --> 10
    with tf.name_scope('layer_fc2'):
        """Softmax层"""
        w_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y_conv = tf.nn.softmax(tf.matmul(dropped, w_fc2) + b_fc2)

    # 计算Cross_entropy
    with tf.name_scope('cross_entropy'):
        diff = y_ * tf.log(y_conv)
    with tf.name_scope('total'):
        cross_entropy = -tf.reduce_mean(diff)
    tf.scalar_summary('cross entropy', cross_entropy)

    with tf.name_scope('train'):
        """AdamOptimizer训练"""
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
    
    with tf.name_scope('accuracy'):
        """计算Accuracy"""
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.scalar_summary('accuracy', accuracy)

    # 汇总概要并将其写入日志文件
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train',sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
    tf.initialize_all_variables().run()

    # 模型训练
    def feed_dict(train):
        """利用Placeholder传入训练数据"""
        if train or FLAGS.fake_data:
            xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
            k = FLAGS.dropout
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}
    
    for i in range(FLAGS.max_steps):
        if i % 10 == 0:
            """记录测试概要以及测试精度"""
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:
            """记录训练概要以及训练精度"""
            if i % 100 == 99:
                """每100次迭代，记录一次训练状态概要信息"""
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:
                """记录概要信息"""
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)

    # 关闭概要写者
    train_writer.close()
    test_writer.close()


def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()


if __name__ == '__main__':
    tf.app.run()

