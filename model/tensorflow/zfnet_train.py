import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *

# Dataset Parameters
batch_size = 180  #Reduce to < 200
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.001
dropout = 0.5 # Dropout, probability to keep units
training_iters = 50000
step_display = 50
step_save = 1
path_save = './zfnet_bn/zfnet_bn'
start_from = ''

def batch_norm_layer(x, train_phase, scope_bn):
    return batch_norm(x, decay=0.9, center=True, scale=True,
    updates_collections=None,
    is_training=train_phase,
    reuse=None,
    trainable=True,
    scope=scope_bn)


weights={
    'conv1': tf.Variable(tf.random_normal([7,7,3,96])),
    'conv2': tf.Variable(tf.random_normal([5,5,96,256])),
    'conv3': tf.Variable(tf.random_normal([3,3,256,384])),
    'conv4': tf.Variable(tf.random_normal([3,3,384,384])),
    'conv5': tf.Variable(tf.random_normal([3,3,384,256])),
    'fc1': tf.Variable(tf.random_normal([7*7*256,4096])),
    'fc2': tf.Variable(tf.random_normal([4096,4096])),
    'fc3': tf.Variable(tf.random_normal([4096,100])),
}

biases={
    'conv1': tf.Variable(tf.random_normal([96])),
    'conv2': tf.Variable(tf.random_normal([256])),
    'conv3': tf.Variable(tf.random_normal([384])),
    'conv4': tf.Variable(tf.random_normal([384])),
    'conv5': tf.Variable(tf.random_normal([256])),
    'fc1': tf.Variable(tf.random_normal([4096])),
    'fc2': tf.Variable(tf.random_normal([4096])),
    'fc3': tf.Variable(tf.random_normal([100]))
}

def conv(self,x,weight,bias,stride):
    return tf.nn.relu(tf.add(tf.nn.conv2d(x,weight,strides=[1,stride,stride,1],padding='SAME'),bias))

def maxpool(self,x,stride,kernelsize):
    return tf.nn.max_pool(x,strides=[1,stride,stride,1],ksize=[1,kernelsize,kernelsize,1],padding='SAME')

def contrastnorm(self,x):
    y=[]
    for item in x:
        y.append(tf.image.per_image_standardization(item))
    return y

def fc(self,x,weight,bias):
    return tf.nn.relu(tf.add(tf.matmul(x,weight),bias))

    
def zfnet(x, keep_dropout, train_phase):

    z=tf.reshape(x,[-1,224,224,3])

    conv1 = conv(z,weights['conv1'],biases['conv1'],2)
    maxpool1=maxpool(conv1,2,3)
    contrastnorm1=contrastnorm(maxpool1)

    conv2=conv(contrastnorm1,weights['conv2'],biases['conv2'],2)
    maxpool2=maxpool(conv2,2,3)
    contrastnorm2=contrastnorm(maxpool2)

    conv3=conv(contrastnorm2,weights['conv3'],biases['conv3'],1)
    maxpool3=maxpool(conv3,1,3)
    contrastnorm3=contrastnorm(maxpool3)

    conv4=conv(contrastnorm3,weights['conv4'],biases['conv4'],1)
    maxpool4=maxpool(conv4,1,3)
    contrastnorm4=contrastnorm(maxpool4)

    conv5=conv(contrastnorm4,weights['conv5'],biases['conv5'],1)
    maxpool5=maxpool(conv5,2,3)
    contrastnorm5=contrastnorm(maxpool5)

    recontrastnorm5=tf.reshape(contrastnorm5,[-1,7*7*256])

    fc1=fc(recontrastnorm5,weights['fc1'],biases['fc1'])
    fc2=fc(fc1,weights['fc2'],biases['fc2'])
    fc3=tf.add(tf.matmul(fc2,weights['fc3']),biases['fc3'])

    return fc3

    # weights = {
    #     'wc1': tf.Variable(tf.random_normal([7, 7, 3, 96], stddev=np.sqrt(2./(7*7*3)))),  # 11x11 -> 7x7 filter first layer
	#     'wc1.5': tf.Variable(tf.random_normal([5, 5, 96, 96], stddev=np.sqrt(2./(5*5*96)))), ##
    #     'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256], stddev=np.sqrt(2./(5*5*96)))),
    #     'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384], stddev=np.sqrt(2./(3*3*256)))),
    #     'wc4': tf.Variable(tf.random_normal([3, 3, 384, 256], stddev=np.sqrt(2./(3*3*384)))),
    #     'wc5': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2./(3*3*256)))),
    # 
    #     'wf6': tf.Variable(tf.random_normal([7*7*256, 4096], stddev=np.sqrt(2./(7*7*256)))),
    #     'wf7': tf.Variable(tf.random_normal([4096, 4096], stddev=np.sqrt(2./4096))),
    #     'wo': tf.Variable(tf.random_normal([4096, 100], stddev=np.sqrt(2./4096)))
    # }
    # 
    # biases = {
    #     'bo': tf.Variable(tf.ones(100))
    # }

    # # Conv + ReLU + Pool, 224->55->27
    # # 224->110->55->27
    # print(x.get_shape()) # [?, 224, 224, 3]
    # conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 2, 2, 1], padding='SAME')  # 4 -> 2 stride
    # print(conv1.get_shape()) # [?, 112, 112, 96]
    # conv1 = tf.nn.conv2d(conv1, weights['wc1.5'], strides=[1, 2, 2, 1], padding="SAME")
    # print(conv1.get_shape())
    # conv1 = batch_norm_layer(conv1, train_phase, 'bn1')
    # conv1 = tf.nn.relu(conv1)
    # pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 
    # # Conv + ReLU  + Pool, 27-> 13
    # conv2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    # conv2 = batch_norm_layer(conv2, train_phase, 'bn2')
    # conv2 = tf.nn.relu(conv2)
    # pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 
    # # Conv + ReLU, 13-> 13
    # conv3 = tf.nn.conv2d(pool2, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
    # conv3 = batch_norm_layer(conv3, train_phase, 'bn3')
    # conv3 = tf.nn.relu(conv3)
    # 
    # # Conv + ReLU, 13-> 13
    # conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=[1, 1, 1, 1], padding='SAME')
    # conv4 = batch_norm_layer(conv4, train_phase, 'bn4')
    # conv4 = tf.nn.relu(conv4)
    # 
    # # Conv + ReLU + Pool, 13->6
    # conv5 = tf.nn.conv2d(conv4, weights['wc5'], strides=[1, 1, 1, 1], padding='SAME')
    # conv5 = batch_norm_layer(conv5, train_phase, 'bn5')
    # conv5 = tf.nn.relu(conv5)
    # pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 
    # # FC + ReLU + Dropout
    # fc6 = tf.reshape(pool5, [-1, weights['wf6'].get_shape().as_list()[0]])
    # fc6 = tf.matmul(fc6, weights['wf6'])
    # fc6 = batch_norm_layer(fc6, train_phase, 'bn6')
    # fc6 = tf.nn.relu(fc6)
    # fc6 = tf.nn.dropout(fc6, keep_dropout)
    # 
    # # FC + ReLU + Dropout
    # fc7 = tf.matmul(fc6, weights['wf7'])
    # fc7 = batch_norm_layer(fc7, train_phase, 'bn7')
    # fc7 = tf.nn.relu(fc7)
    # fc7 = tf.nn.dropout(fc7, keep_dropout)
    # 
    # # Output FC
    # out = tf.add(tf.matmul(fc7, weights['wo']), biases['bo'])
    # print(out.get_shape())
    # return out

# Construct dataloader
opt_data_train = {
    #'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
    }
opt_data_val = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

loader_train = DataLoaderDisk(**opt_data_train)
loader_val = DataLoaderDisk(**opt_data_val)
#loader_train = DataLoaderH5(**opt_data_train)
#loader_val = DataLoaderH5(**opt_data_val)

# tf Graph input
x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
y = tf.placeholder(tf.int64, None)
keep_dropout = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)

# Construct model
logits = zfnet(x, keep_dropout, train_phase)
print(logits.get_shape())
# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Evaluate model
accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 1), tf.float32))
accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 5), tf.float32))

# define initialization
init = tf.global_variables_initializer()

# define saver
saver = tf.train.Saver()

# define summary writer
#writer = tf.train.SummaryWriter('.', graph=tf.get_default_graph())

# Launch the graph
with tf.Session() as sess:
    # Initialization
    step = 0
    if len(start_from)>1:
        saver.restore(sess, start_from)
        step = int(start_from.split('-')[-1])
    else:
        sess.run(init)


    while step < training_iters:
        # Load a batch of training data
        images_batch, labels_batch = loader_train.next_batch(batch_size)

        if step % step_display == 0:
            print('[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            # Calculate batch loss and accuracy on training set
            l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False})
            print("-Iter " + str(step) + ", Training Loss= " + \
                  "{:.6f}".format(l) + ", Accuracy Top1 = " + \
                  "{:.4f}".format(acc1) + ", Top5 = " + \
                  "{:.4f}".format(acc5))

            # Calculate batch loss and accuracy on validation set
            images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)
            l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch_val, y: labels_batch_val, keep_dropout: 1., train_phase: False})
            print("-Iter " + str(step) + ", Validation Loss= " + \
                  "{:.6f}".format(l) + ", Accuracy Top1 = " + \
                  "{:.4f}".format(acc1) + ", Top5 = " + \
                  "{:.4f}".format(acc5))

        # Run optimization op (backprop)
        sess.run(train_optimizer, feed_dict={x: images_batch, y: labels_batch, keep_dropout: dropout, train_phase: True})

        step += 1

        # Save model
        if step % step_save == 0:
            saver.save(sess, path_save, global_step=step)
            print("Model saved at Iter %d !" %(step))

    print("Optimization Finished!")


    # Evaluate on the whole validation set
    print('Evaluation on the whole validation set...')
    num_batch = loader_val.size()//batch_size
    acc1_total = 0.
    acc5_total = 0.
    loader_val.reset()
    for i in range(num_batch):
        images_batch, labels_batch = loader_val.next_batch(batch_size)
        acc1, acc5 = sess.run([accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False})
        acc1_total += acc1
        acc5_total += acc5
        print("Validation Accuracy Top1 = " + \
              "{:.4f}".format(acc1) + ", Top5 = " + \
              "{:.4f}".format(acc5))

    acc1_total /= num_batch
    acc5_total /= num_batch
    print('Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total))