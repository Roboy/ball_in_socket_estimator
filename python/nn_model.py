import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def mean_square_error(y_logit, y_true):
    y_diff = y_true - y_logit
    mse = tf.reduce_sum(tf.square(y_diff), axis=1)

    return tf.reduce_mean(mse)


class NeuralNetworkModel:
    def __init__(self, name, hidden_size=100):

        self.name = name

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.hidden_size = hidden_size

            self.x = tf.placeholder(dtype=tf.float32, shape=[None, 12])
            self.y = tf.placeholder(dtype=tf.float32, shape=[None, 6])
            self.output = self._build()

            learning_rate = 5e-4
            self.mse = mean_square_error(self.output, self.y)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.step_op = optimizer.minimize(self.mse)

            initialization = tf.global_variables_initializer()
            self.saver = tf.train.Saver(max_to_keep=20)

        self.session = tf.Session(graph=self.graph)
        self.session.run(initialization)

    def _build(self):
        hidden = tf.layers.dense(self.x, self.hidden_size, tf.nn.tanh,
                                 kernel_initializer='glorot_uniform', name=self.name+"_layer_1")
        hidden = tf.layers.dense(hidden, self.hidden_size, tf.nn.tanh,
                                 kernel_initializer='glorot_uniform', name=self.name+"_layer_2")
        return tf.layers.dense(hidden, 6, tf.nn.tanh,
                               kernel_initializer='glorot_uniform', name=self.name+"_layer_out")

    def save_model(self, path):
        self.saver.save(self.session, path, write_meta_graph=False)

    def restore_model(self, path):
        self.saver.restore(self.session, path)

    def predict(self, x):
        return self.session.run([self.output], feed_dict={self.x: x})[0]

    def fit(self, x, y, x_val, y_val, save_path, iteration=3000, patience=100, batch_size=1000):

        cur_patience = 0
        best_mse = 999.
        n_train_batches = x.shape[0] // batch_size
        n_valid_batches = x_val.shape[0] // batch_size
        rand_idx = np.arange(x.shape[0])

        for epoch in range(iteration):

            training_loss = 0.
            valid_loss = 0.

            np.random.shuffle(rand_idx)
            rand_in_train = x[rand_idx]
            rand_out_train = y[rand_idx]

            for i in range(n_train_batches - 1):
                idx = slice(i*batch_size, i*batch_size+batch_size)

                minibatch_x = rand_in_train[idx]
                minibatch_y = rand_out_train[idx]

                feed_dict = {
                    self.x: minibatch_x,
                    self.y: minibatch_y,
                }

                _, loss = self.session.run([self.step_op, self.mse], feed_dict)

                training_loss += loss / n_train_batches

            for i in range(n_valid_batches - 1):
                idx = slice(i*batch_size, i*batch_size+batch_size)

                minibatch_x = x_val[idx]
                minibatch_y = y_val[idx]

                feed_dict = {
                    self.x: minibatch_x,
                    self.y: minibatch_y,
                }

                y_pred, loss = self.session.run([self.output, self.mse], feed_dict)

                valid_loss += loss / n_valid_batches

            if valid_loss < best_mse:
                best_mse = valid_loss
                self.save_model(save_path)
                cur_patience = 0
            else:
                cur_patience += 1

            if epoch % 10 == 0:
                print("#%04d: Traing loss %.5f, Valid loss %.5f" % (epoch, training_loss, valid_loss))

            if cur_patience >= patience:
                print("Stop training after %04d iterations" % epoch)
                break
