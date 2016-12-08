# -*- coding: utf-8 -*-
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import re
import warnings
import sys
import time
import numpy as np
import os
import time
import datetime
import random
warnings.filterwarnings("ignore") 
import tensorflow as tf
from tensorflow.contrib import learn
import csv
"""
使用tensorflow进行进行多类别分类
通常，可以使用Logistic regression进行二分类分析
而softmax函数，是Logistic regression在多分类问题上的泛化版本
在这里，依然使用cross_entropy来定义损失函数
注：
    在tensorflow中，多分类的cross_entropy定义有多种，常见的
    2个cross_entropy函数分别为：
                            tf.nn.sparse_softmax_cross_entropy_with_logits
                            tf.nn.softmax_cross_entropy_with_logits
    在多分类问题中，若最终的类别只有一个，如需要告知别人最终类别是A或者属于B或者属于C时，使用tf.nn.sparse_softmax_cross_entropy_with_logits
    在多分类问题中，若最终的类别是一个概率输出，如输入70的概率属于A、20%的概率属于B、10%的概率属于C,则使用tf.nn.softmax_cross_entropy_with_logits
"""
"""
let's define the model
"""
def read_csv():
    reader = csv.reader(file('iris.data', 'rb'))
    next(reader)
    res = []
    lables = []
    for line in reader:
        if len(line) < 5:continue
        sepal_length, sepal_width, petal_length, petal_width, label = line
        class_id = [0,0,0]
        if label=="Iris-setosa":
            class_id = [1,0,0]
        elif label=="ris-versicolor":
            class_id = [0,1,0]
        else:
            class_id = [0,0,1]
        sample = [sepal_length, sepal_width, petal_length, petal_width]
        res.append(sample)
        lables.append(class_id)
    return res,lables

batch_size = 20
res,lables = read_csv()
n_batches = len(res)/batch_size#
n_train_batches = int(np.round(n_batches*0.9))
def get_next_batch(minibatch_index):
    batchs = res[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
    labs = lables[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
    return batchs, labs


#输入的展位符
x_in = tf.placeholder(tf.float32, [None,4],name="input_x")
y_in = tf.placeholder(tf.float32, [None,3],name="input_y")

#待学习的参数
W = tf.Variable(tf.zeros([4,3]), name="weights")
b = tf.Variable(tf.zeros([3]), name="bias")

#function:wx+b
def inference():
    return tf.nn.xw_plus_b(x_in, W, b)
#以cross_entropy定义损失函数
#tf.argmax(y_in,1) 将y_in=[05,0.7,0.3] ==>> 1(最大值的index)
#tf.nn.sparse_softmax_cross_entropy_with_logits(logstic, lables_id): logstic=[0.5,0.9,1.0]这种

l2_reg_lambda = 0.001
def loss():
    l2_loss = tf.nn.l2_loss(W)
    l2_loss += tf.nn.l2_loss(b)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(inference(), tf.to_int32(tf.argmax(y_in,1)))) + l2_reg_lambda * l2_loss

#一次训练
global_step = tf.Variable(0)
def train(total_loss, learning_rate):
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss, global_step)
def accuracy():
    #predicted = tf.cast(tf.arg_max(inference(), 1), tf.int32)
    predicted = tf.equal(tf.argmax(inference(),1), tf.argmax(y_in,1))
    return tf.reduce_mean(tf.cast(predicted, tf.float32))
#输入X并预测Y值
def evaluate():
    predicted = tf.argmax(inference(), 1, name="predicted")
    print tf.reduce_mean(tf.cast(tf.equal(predicted, tf.argmax(y_in,1)), tf.float32))   

total_steps = 500
learning_rate = 0.001
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    total_loss = loss()
    total_loss_summary = tf.scalar_summary(b'total_loss', total_loss, name="total_loss")
    accuracy = accuracy()
    acc_summary = tf.scalar_summary("accuracy", accuracy, name="accuracy")
    train_summary_op = tf.merge_summary([total_loss_summary, acc_summary])
    writer = tf.train.SummaryWriter('./softmax_graph', sess.graph)
    train_step = train(total_loss,learning_rate)
    for i in range(total_steps):
        for minibatch_index in np.random.permutation(range(n_train_batches)):
            X_in, Y_in = get_next_batch(minibatch_index)
            feed_dict={x_in: X_in, y_in: Y_in}
            _, step,  summary = sess.run([train_step, global_step, train_summary_op],feed_dict)
            writer.add_summary(summary, step)
            train_accuracy = accuracy.eval(feed_dict={x_in: res, y_in: lables})
            print "training accuracy:", train_accuracy
    writer.close()










