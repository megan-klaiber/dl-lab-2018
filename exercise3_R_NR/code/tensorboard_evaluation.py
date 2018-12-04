import tensorflow as tf

class Evaluation:

    def __init__(self, store_dir):
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.tf_writer = tf.summary.FileWriter(store_dir)

        self.tf_loss = tf.placeholder(tf.float32, name="loss_summary")
        tf.summary.scalar("loss", self.tf_loss)

        # TODO: define more metrics you want to plot during training (e.g. training/validation accuracy)
        # train accuracy
        self.tf_train_acc = tf.placeholder(tf.float32, name="train_acc_summary")
        tf.summary.scalar("train_acc", self.tf_train_acc)
        # train error
        self.tf_train_err = tf.placeholder(tf.float32, name="train_err_summary")
        tf.summary.scalar("train_err", self.tf_train_err)
        # valid accuracy
        self.tf_valid_acc = tf.placeholder(tf.float32, name="valid_acc_summary")
        tf.summary.scalar("valid_acc", self.tf_valid_acc)
        # valid error
        self.tf_valid_err = tf.placeholder(tf.float32, name="valid_err_summary")
        tf.summary.scalar("valid_err", self.tf_valid_err)
             
        self.performance_summaries = tf.summary.merge_all()

    def write_episode_data(self, episode, eval_dict):

       # TODO: add more metrics to the summary 
       summary = self.sess.run(self.performance_summaries, feed_dict={self.tf_loss : eval_dict["loss"],
                                                                      self.tf_train_err: eval_dict['train_err'], self.tf_train_acc: eval_dict['train_err'],
                                                                      self.tf_valid_err: eval_dict['valid_err'], self.tf_valid_acc: eval_dict['valid_acc']})

       self.tf_writer.add_summary(summary, episode)
       self.tf_writer.flush()

    def close_session(self):
        self.tf_writer.close()
        self.sess.close()
