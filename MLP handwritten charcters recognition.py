# import libraries
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.platform import flags
from utils import *
tf.disable_eager_execution()
# define settings
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_classes', 50, 'number of classes used')
flags.DEFINE_integer('num_samples_train', 15, 'number of samples per class used for training')
flags.DEFINE_integer('num_samples_test', 5, 'number of samples per class used for testing')
flags.DEFINE_integer('seed', 1, 'random seed')


# define you model, loss functions, hyperparameters, and optimizers
### Your Code Here ###

def MLP_add_layer(train_data,input_size,output_size,activation_function=None):

    weights = tf.Variable(tf.truncated_normal([input_size,output_size],mean=0,stddev=0.1,dtype="float32"))
    biases = tf.Variable(tf.zeros([1,output_size],dtype="float32")+0.1)
    input_mul_wb = tf.matmul(train_data,weights) + biases
    if activation_function is None:
        outputs = input_mul_wb
    else:
        outputs = activation_function(input_mul_wb)
    return outputs




# load data
train_image, train_label, test_image, test_label = LoadData(FLAGS.num_classes, FLAGS.num_samples_train,
                                                            FLAGS.num_samples_test, FLAGS.seed)

# note: you should use train_image, train_label for training, apply the model to test_image to get predictions and use test_label to evaluate the predictions
test_label = tf.keras.utils.to_categorical(test_label)
train_label = tf.keras.utils.to_categorical(train_label)

x_zdy = tf.placeholder(tf.float32,[None,784])
y_zdy = tf.placeholder(tf.float32)

mlp_l1 = MLP_add_layer(x_zdy, 784, 300, activation_function=tf.nn.relu)
mlp_l2 = MLP_add_layer(mlp_l1, 300, 128, activation_function=tf.nn.relu)
mlp_pred = MLP_add_layer(mlp_l2, 128, 50, activation_function=tf.nn.softmax)

loss = tf.reduce_mean(tf.reduce_sum((mlp_pred - y_zdy)**2))

train_step = tf.train.AdamOptimizer(0.0003).minimize(loss)

loss_record = []
accuracy_record = []

with tf.Session() as sess:
    # initialize
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    # train model using train_image and train_label
    ### Your Code Here ###
    for i in range(5000):
        sess.run(train_step,feed_dict={x_zdy:train_image,y_zdy:train_label})

        if i%50 == 0:
            loss_run = sess.run(loss, feed_dict={x_zdy: train_image, y_zdy: train_label})
            print(loss_run)
            loss_record.append(loss_run)
            find_equal = tf.equal(tf.arg_max(y_zdy, 1), tf.arg_max(mlp_pred, 1))
            pred = tf.reduce_mean(tf.cast(find_equal, tf.float32))
            accu_pred = sess.run(pred, feed_dict={x_zdy: test_image, y_zdy: test_label})
            print("Test Accuracy:", accu_pred)
            accuracy_record.append(accu_pred)
    # get predictions on test_image
    ### Your Code Here ###

    find_equal = tf.equal(tf.arg_max(y_zdy,1),tf.arg_max(mlp_pred,1))
    pred = tf.reduce_mean(tf.cast(find_equal,tf.float32))

    # evaluation
    print("Traning End.Test Accuracy:", sess.run(pred,feed_dict={x_zdy:test_image,y_zdy:test_label}))
    print("loss = ",loss_record)
    print("accuracy = ",accuracy_record)
    # note that you should not use test_label elsewhere





