import tensorflow as tf
import numpy as np

class cnn:

    # constructor
    def __init__(self, num_epochs, lr, num_filters, batch_size, filter_size, model_name):
        self.num_epochs = num_epochs
        self.lr = lr
        self.num_filters = num_filters
        self.batch_size = batch_size
        self.filter_size = filter_size
        self.model_name = model_name

        # placeholder
        self.x_placeholder = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        self.y_placeholder = tf.placeholder(tf.float32, shape=[None, 10])

    def network_graph(self, x):
        ###### network ######
        # first conv layer
        conv1 = tf.layers.conv2d(
            inputs=x,
            filters=self.num_filters,
            kernel_size=self.filter_size,
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )

        # first pooling layer
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=1)

        # second conv layer
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=self.num_filters,
            kernel_size=self.filter_size,
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )

        # second pooling layer
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=1)

        # flatten
        flatten_pool2 = tf.contrib.layers.flatten(pool2)

        # fully connected layer
        full = tf.contrib.layers.fully_connected(
            inputs=flatten_pool2,
            num_outputs=128
        )

        # output layer with softmax
        logits = tf.layers.dense(inputs=full, units=10)
        #####################

        return logits

    def train(self, x_train, y_train, x_valid, y_valid):

        # build network graph
        logits = self.network_graph(self.x_placeholder)

        ###### sgd ######
        # calculate training loss
        soft_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_placeholder, logits=logits)
        loss = tf.reduce_mean(soft_entropy)
        # optimizer -> stochastic gradient descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        training_op = optimizer.minimize(loss)
        #################

        # number of batches
        n_samples = x_train.shape[0]
        n_batches = n_samples // self.batch_size

        # number of batches valid
        n_samples_valid = x_valid.shape[0]
        n_batches_valid = n_samples_valid // self.batch_size

        # saver
        saver = tf.train.Saver()
        save_path = './model/' + self.model_name

        # accuracies
        train_loss = np.zeros((self.num_epochs))
        #train_accuracy = np.zeros((self.num_epochs))
        valid_accuracy = np.zeros((self.num_epochs))
        valid_error = np.zeros((self.num_epochs))

        # correct = tf.nn.in_top_k(logits, y_placeholder, 1)
        # accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        # calculate the accuracy
        y_pred = tf.argmax(logits, 1)
        tf.add_to_collection('pred_network', y_pred)
        correct_pred = tf.equal(y_pred, tf.argmax(self.y_placeholder, 1))
        # calculate train and valid accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # iterate over epochs
            for e in range(self.num_epochs):
                # iterate over batches
                for b in range(n_batches):
                    # extracting a batch from x_train and y_train
                    start = b * self.batch_size
                    end = start + self.batch_size
                    x_batch = x_train[start:end, ]
                    y_batch = y_train[start:end, ]

                    # train
                    _, temp_loss = sess.run([training_op, loss],
                                            feed_dict={self.x_placeholder: x_batch, self.y_placeholder: y_batch})
                    train_loss[e] += (temp_loss / self.batch_size)

                # save accuracies
                #train_accuracy[e] = accuracy.eval({self.x_placeholder: x_train, self.y_placeholder: y_train})
                #valid_accuracy[e] = accuracy.eval({self.x_placeholder: x_valid, self.y_placeholder: y_valid})
                #valid_error[e] = 1 - valid_accuracy[e]

                for b in range(n_batches_valid):
                    # extracting a batch from x_train and y_train
                    start = b * self.batch_size
                    end = start + self.batch_size
                    x_batch_valid = x_valid[start:end, ]
                    y_batch_valid = y_valid[start:end, ]
                    valid_accuracy[e] += accuracy.eval({self.x_placeholder: x_batch_valid, self.y_placeholder: y_batch_valid})

                valid_accuracy[e] = valid_accuracy[e] / n_batches_valid
                valid_error[e] = 1 - valid_accuracy[e]

                print("[%d/%d]: valid_accuracy: %.4f, valid_error: %.4f" % (e + 1, self.num_epochs, valid_accuracy[e], valid_error[e]))

            # save model
            saver.save(sess, save_path)
            print("Model saved in path: %s" % save_path)

        return train_loss, valid_accuracy, valid_error


    def test(self, x_test, y_test):
        # saver = tf.train.Saver()
        save_path = "./model/" + self.model_name + ".meta"
        saver = tf.train.import_meta_graph(save_path)

        n_batches = x_test.shape[0] // self.batch_size
        test_error = 0

        with tf.Session() as sess:
            # saver.restore(sess, model)
            saver.restore(sess, tf.train.latest_checkpoint("./model"))
            print('Model restored.')

            y_pred = tf.get_collection("pred_network")[0]

            # prediction = np.array(sess.run(y_pred, feed_dict={x_placeholder: x_test, y_placeholder: y_test}))
            #prediction = np.array(sess.run(y_pred, feed_dict={self.x_placeholder: x_test}))
            #test_error = float(np.sum(prediction != np.argmax(y_test, axis=1)) / y_test.shape[0])

            for b in range(n_batches):
                start = b * self.batch_size
                end = start + self.batch_size
                x_batch = x_test[start:end, ]
                y_batch = y_test[start:end, ]

                prediction = np.array(sess.run(y_pred, feed_dict={self.x_placeholder: x_batch}))
                test_error += float(np.sum(prediction != np.argmax(y_test, axis=1))) / (self.batch_size * n_batches)

            print("test error: %.4f" % test_error)

        return test_error
