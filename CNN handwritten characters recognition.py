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

def add_cnn_layer(train_data,filter_height,filter_width,step,
                  channel, activation_function=None, norm=False):
    Weights = tf.Variable(tf.ones([filter_height,filter_width
                                               ,step,channel],dtype="float32"))
    bias = tf.Variable(tf.constant(0.1,shape=[channel],dtype="float32"))
    conv2d_layer = tf.nn.conv2d(train_data,Weights,strides=[1,1,1,1],padding="VALID") + bias

    if norm :
        fc_mean, fc_var = tf.nn.moments(
            conv2d_layer, axes=[0,1,2]
        )

        scale = tf.Variable(tf.ones([channel]))
        shift = tf.Variable(tf.zeros([channel]))
        epsilon = 0.001
        moving_average = tf.train.ExponentialMovingAverage(decay=0.5)
        def mean_var_updata_cnn():
            moving_average_op = moving_average.apply([fc_mean,fc_var])
            with tf.control_dependencies([moving_average_op]):
                return tf.identity(fc_mean),tf.identity(fc_var)
        mean, var = mean_var_updata_cnn()
        conv2d_layer = tf.nn.batch_normalization(conv2d_layer,mean,var,shift,
                                             scale,epsilon)


    if activation_function is None:
        outputs = tf.nn.max_pool(conv2d_layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

    else:
        outputs = tf.nn.max_pool(activation_function(conv2d_layer),ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

    return outputs

def add_mlp_layer(train_data,input_size,output_size,activation_function=None,norm = False):

    weights = tf.Variable(tf.random.truncated_normal([input_size,output_size],
                                             mean=0,stddev=0.1,dtype="float32"))
    biases = tf.Variable(tf.zeros([1,output_size],dtype="float32")+0.1)
    input_mul_wb = tf.matmul(train_data,weights) + biases
    if norm:

        fc_mean, fc_var = tf.nn.moments(
            input_mul_wb,
            axes=[0],
        )
        scale = tf.Variable(tf.ones([output_size]))
        shift = tf.Variable(tf.zeros([output_size]))
        epsilon = 0.001

        # apply moving average for mean and var when train on batch
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([fc_mean, fc_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(fc_mean), tf.identity(fc_var)

        mean, var = mean_var_with_update()

        input_mul_wb = tf.nn.batch_normalization(input_mul_wb, mean, var, shift, scale, epsilon)


    if activation_function is None:
        outputs = input_mul_wb
    else:
        outputs = activation_function(input_mul_wb)
    return outputs



# load data
train_image, train_label, test_image, test_label = LoadData(FLAGS.num_classes, FLAGS.num_samples_train,
                                                            FLAGS.num_samples_test, FLAGS.seed)

# note: you should use train_image, train_label for training, apply the model to test_image to get predictions and use test_label to evaluate the predictions
train_image = np.reshape(train_image,[-1,28,28,1])
test_image = np.reshape(test_image,[-1,28,28,1])
test_label = tf.keras.utils.to_categorical(test_label)
train_label = tf.keras.utils.to_categorical(train_label)


x_zdy = tf.placeholder(tf.float32)
y_zdy = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

###conv1 layer####
cnn_layer_1 = add_cnn_layer(x_zdy,3,3,1,32,activation_function=tf.nn.relu,norm=True)
###conv2 layer####
cnn_layer_2 = add_cnn_layer(cnn_layer_1,4,4,32,64,activation_function=tf.nn.relu,norm=True)
###func1 layer####
cnn_layer_3_flat = tf.reshape(cnn_layer_2,[-1,5*5*64])
func_layer_1 = add_mlp_layer(cnn_layer_3_flat,5*5*64,512,activation_function=tf.nn.relu,norm=True)

###func2 layer####
func_layer_2 = add_mlp_layer(func_layer_1,512,50,activation_function=tf.nn.softmax,norm=False)
###################

loss = tf.reduce_mean(-tf.reduce_sum(y_zdy*tf.log(func_layer_2),
                                     reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.0005).minimize(loss)


loss_record = []
accuracy_record = []

with tf.Session() as sess:
    # initialize
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    # train model using train_image and train_label
    ### Your Code Here ###
    for i in range(4000):
        sess.run(train_step,feed_dict={x_zdy:train_image,y_zdy:train_label,keep_prob:[0.76]})

        if i%50 == 0:
            loss_run = sess.run(loss, feed_dict={x_zdy: train_image, y_zdy: train_label, keep_prob: [0.76]})
            print(loss_run)
            loss_record.append(loss_run)
            find_equal = tf.equal(tf.arg_max(y_zdy, 1), tf.arg_max(func_layer_2, 1))
            pred = tf.reduce_mean(tf.cast(find_equal, tf.float32))
            accuracy_run = sess.run(pred, feed_dict={x_zdy: test_image, y_zdy: test_label})
            print("Test Accuracy:", accuracy_run)
            accuracy_record.append(accuracy_run)
    # get predictions on test_image
    ### Your Code Here ###

    find_equal = tf.equal(tf.arg_max(y_zdy,1),tf.arg_max(func_layer_2,1))
    pred = tf.reduce_mean(tf.cast(find_equal,tf.float32))

    # evaluation
    print("Training end, Test Accuracy:", sess.run(pred,feed_dict={x_zdy:test_image,y_zdy:test_label}))
    print(loss_record)
    print(accuracy_record)
    # note that you should not use test_label elsewhere





