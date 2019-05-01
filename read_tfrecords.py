# -*- coding: utf-8 -*-
"""

读取TFRecords，将其打乱顺序批量输出

"""

##################### load packages #####################
import tensorflow as tf

class get_data():
    def __init__(self, filenames, data_size):

        self.filenames = filenames
        self.data_size = data_size

    ##################### 解析tfrecords ######################
    def parse_data(self, example_proto):
        features = {'img': tf.FixedLenFeature([], tf.string, ''),
                    'label': tf.FixedLenFeature([], tf.int64, 0)}
        parsed_features = tf.parse_single_example(example_proto, features)
        image = tf.decode_raw(parsed_features['img'], tf.uint8)
        label = tf.cast(parsed_features['label'], tf.int64)
        image = tf.reshape(image, [500, 500, 3])
        return image, label

    ##################### 输入数据流 ######################
    def my_input_fn(self, filenames, data_size):
        dataset = tf.contrib.data.TFRecordDataset(filenames)
        dataset = dataset.map(self.parse_data)
        dataset = dataset.batch(data_size)

        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels


    def main(self):

        features, labels = self.my_input_fn(self.filenames, self.data_size)

        with tf.Session() as sess:
            ########### 初始化 ###########
            init = tf.global_variables_initializer()
            sess.run(init)

            ######### 取出img_batch and label_batch #########
            img, label = sess.run([features, labels])

        return img, label
