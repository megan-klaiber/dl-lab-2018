import tensorflow as tf

class Model:
    
    def __init__(self, lr, num_filters, batch_size, history_length, filter_size=5):
        
        # TODO: Define network
        self.lr = lr
        self.num_filters = num_filters
        self.batch_size = batch_size
        self.filter_size = filter_size
        self.history_length = history_length

        # placeholder
        self.x_placeholder = tf.placeholder(tf.float32, shape=[None, 96, 96, self.history_length])
        self.y_placeholder = tf.placeholder(tf.float32, shape=[None, 9])

        # first conv layer
        self.conv1 = tf.layers.conv2d(
            inputs=self.x_placeholder,
            filters=self.num_filters,
            kernel_size=self.filter_size,
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )

        # first pooling layer
        self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=2, strides=1)

        # second conv layer
        self.conv2 = tf.layers.conv2d(
            inputs=self.pool1,
            filters=self.num_filters,
            kernel_size=self.filter_size,
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )

        # second pooling layer
        self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2, pool_size=2, strides=1)

        # flatten
        self.flatten_pool2 = tf.contrib.layers.flatten(self.pool2)

        # fully connected layer
        self.full = tf.contrib.layers.fully_connected(
            inputs=self.flatten_pool2,
            num_outputs=128
        )

        # output layer
        self.logits = tf.layers.dense(inputs=self.full, units=9)

        # TODO: Loss and optimizer
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_placeholder, logits=self.logits))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)

        # calculate the accuracy
        self.y_pred = tf.argmax(self.logits, 1)
        self.correct_pred = tf.equal(self.y_pred, tf.argmax(self.y_placeholder, 1))
        # calculate train and valid accuracy
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, "float"))

        # TODO: Start tensorflow session
        self.sess = tf.Session()

        self.saver = tf.train.Saver()

    def load(self, file_name):
        self.saver.restore(self.sess, file_name)

    def save(self, file_name):
        self.saver.save(self.sess, file_name)
