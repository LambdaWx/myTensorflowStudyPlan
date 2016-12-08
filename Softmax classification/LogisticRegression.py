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
使用tensorflow进行逻辑回归(Logistic regression)
输入X=(x1,x2....,xn)
输出Y=(y)
映射函数关系：
            Y=1/(1+e^(WX+B))
W为n大小的权重矩阵
B为偏置向量
"""
"""
let's define the model
"""
"""
#从https://www.kaggle.com/c/titanic/data下载traing dataset 
def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer(["/home/myTensorflowStudyPlan/" + file_name])
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)
    # decode_csv will convert a Tensor from type string (the text line) in
    # a tuple of tensor columns with the specified defaults, which also
    # sets the data type for each column
    decoded = tf.decode_csv(value, record_defaults=record_defaults)
    # batch actually reads the file and loads "batch_size" rows in a single tensor
    return tf.train.shuffle_batch(decoded,batch_size=batch_size,capacity=batch_size * 50,min_after_dequeue=batch_size)
def inputs():
    passenger_id, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked = \
    read_csv(100, "train.csv", [[0.0], [0.0], [0], [""], [""], [0.0], [0.0], [0.0], [""], [0.0],[""],[""]])
    # convert categorical data
    # tf.equal(): convert data to bool
    # tf.to_float(): convert bool data to float
    is_first_class = tf.to_float(tf.equal(pclass, [1]))
    is_second_class = tf.to_float(tf.equal(pclass, [2]))
    is_third_class = tf.to_float(tf.equal(pclass, [3]))
    gender = tf.to_float(tf.equal(sex, ["female"]))
    # Finally we pack all the features in a single matrix;
    # We then transpose to have a matrix with one example per row and one feature per column.
    features = tf.transpose(tf.pack([is_first_class, is_second_class, is_third_class, gender, age]))
    survived = tf.reshape(survived, [100, 1])
    features_return = features.eval()
    survived_return = survived.eval()
    return features_return, survived_return
"""
def read_csv():
    reader = csv.reader(file('train.csv', 'rb'))
    next(reader)
    res = []
    lables = []
    for line in reader:
        passenger_id, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked = line
        is_first_class = 0.0
        is_second_class = 0.0
        is_third_class = 0.0
        gender = 0.0
        if pclass=="1":
            is_first_class = 1.0
        elif pclass=="2":
            is_second_class = 1.0
        else:
            is_third_class = 1.0
        if sex=="female":
            gender = 1.0
        sample = [is_first_class, is_second_class, is_third_class, gender, float(age) if age>"0" else 0.0]
        res.append(sample)
        lables.append([int(survived)])
    return res,lables

batch_size = 50
res,lables = read_csv()
n_batches = len(res)/batch_size#
n_train_batches = int(np.round(n_batches*0.9))
def get_next_batch(minibatch_index):
    batchs = res[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
    labs = lables[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
    return batchs, labs


#输入的展位符
x_in = tf.placeholder(tf.float32, [None,5],name="input_x")
y_in = tf.placeholder(tf.float32, [None,1],name="input_y")

#待学习的参数
W = tf.Variable(tf.zeros([5,1]), name="weights")
b = tf.Variable(0., name="bias")

#function:wx+b
def inference():
    return tf.sigmoid(tf.matmul(x_in, W) + b)
#以cross_entropy定义损失函数
l2_reg_lambda = 0.001
def loss():
    l2_loss = tf.nn.l2_loss(W)
    l2_loss += tf.nn.l2_loss(b)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.matmul(x_in, W) + b, y_in)) + l2_reg_lambda * l2_loss
#一次训练
global_step = tf.Variable(0)
def train(total_loss, learning_rate):
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss, global_step)
def accuracy():
    predicted = tf.cast(inference() > 0.5, tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(predicted, y_in), tf.float32))
#输入X并预测Y值
def evaluate(sess):
    predicted = tf.cast(inference() > 0.5, tf.float32)
    print sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, y_in), tf.float32)))

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
    writer = tf.train.SummaryWriter('./LogisticRegression_graph', sess.graph)
    train_step = train(total_loss,learning_rate)
    for i in range(total_steps):
        for minibatch_index in np.random.permutation(range(n_train_batches)):
            X_in, Y_in = get_next_batch(minibatch_index)
            feed_dict={x_in: X_in, y_in: Y_in}
            _, step,  summary = sess.run([train_step, global_step, train_summary_op],feed_dict)
            writer.add_summary(summary, step)
    writer.close()










