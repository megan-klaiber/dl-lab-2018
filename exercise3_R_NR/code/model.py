import tensorflow as tf

class Model:
    
    def __init__(self, lr, num_filters, batch_size, history_length, filter_size=5, optimizer='Adam'):
        
        # TODO: Define network
        self.lr = lr
        self.num_filters = num_filters
        self.batch_size = batch_size
        self.filter_size = filter_size
        self.history_length = history_length

        # placeholder
        self.x_placeholder = tf.placeholder(tf.float32, shape=[None, 96, 96, self.history_length])
        self.y_placeholder = tf.placeholder(tf.float32, shape=[None, 9])

        # first layers + relu
        self.W_conv1 = tf.get_variable("W_conv1", [8, 8, history_length, num_filters],
                                       initializer=tf.contrib.layers.xavier_initializer())
        conv1 = tf.nn.conv2d(self.x_placeholder, self.W_conv1, strides=[1, 2, 2, 1], padding='VALID')
        conv1_a = tf.nn.relu(conv1)
        # second layer + relu:
        self.W_conv2 = tf.get_variable("W_conv2", [4, 4, num_filters, num_filters],
                                       initializer=tf.contrib.layers.xavier_initializer())
        conv2 = tf.nn.conv2d(conv1_a, self.W_conv2, strides=[1, 2, 2, 1], padding='VALID')
        conv2_a = tf.nn.relu(conv2)
        # third layer + relu:
        self.W_conv3 = tf.get_variable("W_conv3", [3, 3, num_filters, num_filters],
                                       initializer=tf.contrib.layers.xavier_initializer())
        conv3 = tf.nn.conv2d(conv2_a, self.W_conv3, strides=[1, 2, 2, 1], padding='VALID')
        conv3_a = tf.nn.relu(conv3)
        # forth layer + relu:
        self.W_conv4 = tf.get_variable("W_conv4", [3, 3, num_filters, 32],
                                       initializer=tf.contrib.layers.xavier_initializer())
        conv4 = tf.nn.conv2d(conv3_a, self.W_conv4, strides=[1, 2, 2, 1], padding='VALID')
        conv4_a = tf.nn.relu(conv4)

        flatten = tf.contrib.layers.flatten(conv4_a)
        # first dense layer + relu + dropout
        fc1 = tf.contrib.layers.fully_connected(flatten, 400, activation_fn=tf.nn.relu)
        fc1_drop = tf.nn.dropout(fc1, 0.8)
        # second dense layer + relu:
        fc2 = tf.contrib.layers.fully_connected(fc1_drop, 400, activation_fn=tf.nn.relu)
        fc2_drop = tf.nn.dropout(fc2, 0.8)
        # third dense layer + relu
        fc3 = tf.contrib.layers.fully_connected(fc2_drop, 50, activation_fn=tf.nn.relu)

        # output layer:
        self.logits= tf.contrib.layers.fully_connected(fc3, 9, activation_fn=None)

        # TODO: Loss and optimizer
        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.y_placeholder, predictions=self.logits))

        if optimizer == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        elif optimizer == 'SGD':
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
