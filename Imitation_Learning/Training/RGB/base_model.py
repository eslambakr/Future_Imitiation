import tensorflow as tf


class BaseModel:
    def __init__(self, config):
        self.config = config
        self.summaries = None
        # global tensors
        self.cur_epoch_tensor = None
        self.cur_epoch_input = None
        self.cur_epoch_assign_op = None
        self.global_step_tensor = None
        # init the global step, global time step, the current epoch and the summaries
        self.init_global_step()
        self.init_cur_epoch()

    def init_saver(self):
        raise NotImplemented

    def save(self, sess):
        print("\nSaving model to {}...".format(self.config.checkpoint_dir))
        sess.run(self.global_step_tensor)
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
        print("\nModel saved")

    def load(self, sess, load=False):
        """Load from last checkpoint"""
        if load:
            latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
            if latest_checkpoint:
                print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
                self.saver.restore(sess, latest_checkpoint)
                print("Model loaded")
            else:
                print("Model not Loaded")
        else:
            init = tf.initialize_all_variables()
            sess.run(init)
            print("Model will start from scratch")

    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    def init_global_step(self):
        """
        Create a global step variable to be a reference to the number of iterations
        :return:
        """
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.increment_global_step_tensor = tf.assign(self.global_step_tensor, self.global_step_tensor + 1)


    def build_model(self):
        raise NotImplementedError
