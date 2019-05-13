import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./mnist/data',one_hot=True)
X = tf.placeholder(tf.float32,[None,784])
Y = tf.placeholder(tf.float32,[None,10])
keep_prop = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([784,256],stddev=0.01))
b1 = tf.Variable(tf.random_normal([256],stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X,W1)+b1)
L1 = tf.nn.dropout(L1,keep_prop)

W2 = tf.Variable(tf.random_normal([256,256],stddev=0.01))
b2 = tf.Variable(tf.random_normal([256],stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1,W2)+b2)
L2 = tf.nn.dropout(L2,keep_prop)

W3 = tf.Variable(tf.random_normal([256,10],stddev=0.01))
model = tf.matmul(L2,W3)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model,labels=Y))
optimizer = tf.train.AdamOptimizer(0.01)
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    batch_size = 100
    total_batch = int(mnist.train.num_examples / batch_size)
    for epoch in range(30):
        total_cost = 0
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            _,cost_val = sess.run([train_op,cost],feed_dict={X:batch_xs,Y:batch_ys,keep_prop:0.8})
            total_cost += cost_val
        print('epoch : %4d'%(epoch+1))
        print('avg cost : (%.3f)'%(total_cost/total_batch))
    print('Done to optimize')

    is_correct = tf.equal(tf.argmax(model,axis=1),tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))
    print('정확도',sess.run(accuracy,feed_dict={X:mnist.test.images,Y:mnist.test.labels,keep_prop:1}))


    