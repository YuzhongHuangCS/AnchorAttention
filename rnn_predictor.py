import dateutil.parser
import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model
import sklearn.preprocessing
import tensorflow as tf
import math

# Each instance is created for each incoming request
class RNNPredictor(object):
    """docstring for RNNPredictor"""

    def __init__(self, config):
        super(RNNPredictor, self).__init__()
        self.config = config
        self.content = None
        self.data = None
        self.data_all = None
        self.dates = None
        self.n_train = None
        self.mse = None
        self.mse_train = None
        self.mse_valid = None
        self.basename = None
        self.pred_train = None
        self.pred_train_lower = None
        self.pred_train_upper = None
        self.pred_valid = None
        self.pred_valid_lower = None
        self.pred_valid_upper = None
        self.pred_test = None
        self.pred_test_lower = None
        self.pred_test_upper = None

    def predict(self, content):
        array = content['payload']['historical_data']['ts']
        data_all = [t[1] for t in array]

        if self.config.test_split:
            data = data_all[:-self.config.n_predict_step]
        else:
            data = data_all

        basename = 'andy_input_{}.json'.format(content['ifp']['id'])
        dates = [dateutil.parser.parse(t[0]) for t in array]
        end_date = dateutil.parser.parse(content['ifp']['ends_at'])
        n_predict_step = (end_date-dates[-1]).days + 1
        self.config.n_predict_step = n_predict_step
        self.config.n_output_dim = n_predict_step * 3
        print('Predict step: ', self.config.n_predict_step, self.config.n_output_dim)

        scaler = sklearn.preprocessing.StandardScaler()
        data_scaled = scaler.fit_transform(np.asarray(data).reshape(-1, 1))

        df = pd.DataFrame(data_scaled)
        df_array = [df.shift(1), df]
        for i in range(-1, -self.config.n_predict_step, -1):
            df_array.append(df.shift(i))
        df = pd.concat(df_array, axis=1)
        df.fillna(value=0, inplace=True)
        x, y = df.values[:, 0], df.values[:, 1:]
        #x = x.reshape(-1, self.config.n_timestep, self.config.n_input_dim)
        #y = np.repeat(y, 3, axis=0).reshape(len(x), self.config.n_output_dim)

        n_valid = max(min(int(len(y) * self.config.ratio_valid), self.config.max_valid), 1)
        n_train = int(int((len(y) - n_valid) / self.config.n_batch) * self.config.n_batch)
        x_train, y_train = x[:n_train], y[:n_train]
        x_valid, y_valid = x[n_train:], y[n_train:]

        #if self.config.n_predict_step != 1:
        #    y_train = [a.reshape(-1, 1, 1) for a in np.hsplit(y_train, self.config.n_output_dim)]
        #    y_valid = [a.reshape(-1, 1, 1) for a in np.hsplit(y_valid, self.config.n_output_dim)]
        #else:
        #    y_train = y_train.reshape(-1, 1, 1)
        #    y_valid = y_valid.reshape(-1, 1, 1)

        num_batches = len(x_train) // self.config.n_batch // self.config.n_timestep

        # build network
        batchX_placeholder = tf.placeholder(tf.float32, [self.config.n_batch, self.config.n_input_dim])
        batchY_placeholder = tf.placeholder(tf.float32, [self.config.n_batch, self.config.n_predict_step])

        cell_state = tf.placeholder(tf.float32, [self.config.n_batch, self.config.n_neurons])

        limit = math.sqrt(6/(self.config.n_neurons + self.config.n_output_dim))
        W2 = tf.Variable(np.random.uniform(-limit, limit, (self.config.n_neurons, self.config.n_output_dim)).astype(np.float32))
        b2 = tf.Variable(np.zeros((1, self.config.n_output_dim)), dtype=tf.float32)

        # Unpack columns
        inputs_series = tf.split(batchX_placeholder, self.config.n_timestep, 0)
        labels_series = batchY_placeholder

        # Forward passes
        cell = tf.nn.rnn_cell.GRUCell(self.config.n_neurons)
        # gru's output is same as state
        states_series, current_state = tf.nn.static_rnn(cell, inputs_series, cell_state)

        prediction = tf.matmul(current_state, W2) + b2
        pred_point = tf.slice(prediction, (0, 0), (1, self.config.n_predict_step))
        pred_lower = tf.slice(prediction, (0, self.config.n_predict_step), (1, self.config.n_predict_step))
        pred_upper = tf.slice(prediction, (0, 2*self.config.n_predict_step), (1, self.config.n_predict_step))

        point_loss = tf.reduce_mean(tf.squared_difference(pred_point, labels_series), axis=-1)

        diff_lower = pred_lower - labels_series
        diff_p_l = tf.reduce_mean(tf.square(tf.clip_by_value(diff_lower, 0, 1e10)), axis=-1)
        diff_n_l = tf.reduce_mean(tf.square(tf.clip_by_value(diff_lower, -1e10, 0)), axis=-1)
        lower_loss = diff_p_l * 1.95 + diff_n_l * 0.05

        diff_upper = pred_upper - labels_series
        diff_p_u = tf.reduce_mean(tf.square(tf.clip_by_value(diff_upper, 0, 1e10)), axis=-1)
        diff_n_u = tf.reduce_mean(tf.square(tf.clip_by_value(diff_upper, -1e10, 0)), axis=-1)
        upper_loss = diff_p_u * 0.05 + diff_n_u * 1.95

        total_loss = point_loss + lower_loss + upper_loss

        learning_rate = tf.Variable(self.config.lr, trainable=False)
        learning_rate_decay_op = learning_rate.assign(learning_rate * self.config.lr_decay)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_step = optimizer.minimize(loss=total_loss)

        saver = tf.train.Saver()
        save_path = self.config.model_prefix + basename.replace('json', 'ckpt')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # make valid model
            smallest_loss = float('inf')
            wait = 0

            for i in range(self.config.n_max_epoch):
                print('Epoch: {}/{}'.format(i, self.config.n_max_epoch))
                # train
                _current_cell_state = np.zeros((self.config.n_batch, self.config.n_neurons), dtype=np.float32)
                train_loss_list = []
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * self.config.n_timestep
                    end_idx = start_idx + self.config.n_timestep

                    batchX = x_train[start_idx:end_idx].reshape(-1, 1)
                    batchY = y_train[start_idx:end_idx]

                    _total_loss, _train_step, _current_state = sess.run(
                        [total_loss, train_step, current_state],
                        feed_dict={
                            batchX_placeholder: batchX,
                            batchY_placeholder: batchY,
                            cell_state: _current_cell_state,
                        })

                    _current_cell_state = _current_state

                    train_loss_list.append(_total_loss)

                train_loss = np.mean(train_loss_list)

                # valid, re use state
                valid_loss_list = []
                for batch_idx in range(len(x_valid)):
                    start_idx = batch_idx
                    end_idx = start_idx+1

                    batchX = x_valid[start_idx:end_idx].reshape(-1, 1)
                    batchY = y_valid[start_idx:end_idx]


                    _total_loss, _current_state = sess.run(
                        [total_loss, current_state],
                        feed_dict={
                            batchX_placeholder: batchX,
                            batchY_placeholder: batchY,
                            cell_state: _current_cell_state,
                        })

                    _current_cell_state = _current_state

                    valid_loss_list.append(_total_loss)

                valid_loss = np.mean(valid_loss_list)
                sum_loss = train_loss * (1 - self.config.valid_loss_weight) + valid_loss * self.config.valid_loss_weight
                print("Epoch", i, "Train Loss", train_loss, 'Valid Loss', valid_loss, 'Sum Loss', sum_loss)
                if wait <= self.config.n_patience:
                    if sum_loss < smallest_loss:
                        smallest_loss = sum_loss
                        saver.save(sess, save_path)
                        wait = 0
                        print('New smallest')
                    else:
                        wait += 1
                        print('Wait {}'.format(wait))
                        if wait % self.config.n_lr_decay == 0:
                            sess.run(learning_rate_decay_op)
                            print('Apply lr decay, new lr: %f' % learning_rate.eval())
                else:
                    break

            print('In test')
            #test
            saver.restore(sess, save_path)
            _current_cell_state = np.zeros((self.config.n_batch, self.config.n_neurons), dtype=np.float32)

            data_forcast = np.insert(data_scaled, 0, 0)
            preds = []
            for batch_idx in range(len(data_forcast)):
                start_idx = batch_idx
                end_idx = start_idx + 1

                batchX = data_forcast[start_idx:end_idx].reshape(-1, 1)

                _prediction, _current_state = sess.run(
                    [prediction, current_state],
                    feed_dict={
                        batchX_placeholder: batchX,
                        cell_state: _current_cell_state,
                    })
                _current_cell_state = _current_state

                preds.append(_prediction)

            pred = np.asarray(preds).squeeze()

            mse = sklearn.metrics.mean_squared_error(data_scaled[n_train:], pred[n_train:-1, 0])

            pred = scaler.inverse_transform(pred)
            pred_train = pred[:n_train, 0]
            pred_valid = pred[n_train:-1, 0]
            pred_train_lower = pred[:n_train, self.config.n_predict_step]
            pred_train_upper = pred[:n_train, 2*self.config.n_predict_step]
            pred_valid_lower = pred[n_train:-1, self.config.n_predict_step]
            pred_valid_upper = pred[n_train:-1, 2*self.config.n_predict_step]

            pred_test = pred[-1, :self.config.n_predict_step]
            pred_test_lower = pred[-1, self.config.n_predict_step:2*self.config.n_predict_step]
            pred_test_upper = pred[-1, 2*self.config.n_predict_step:]

            #pred_train_lower = np.minimum(pred_train, pred_train_lower)
            #pred_train_upper = np.maximum(pred_train, pred_train_upper)
            #pred_valid_lower = np.minimum(pred_valid, pred_valid_lower)
            #pred_valid_upper = np.maximum(pred_valid, pred_valid_upper)
            #pred_test_lower = np.minimum(pred_test, pred_test_lower)
            #pred_test_upper = np.maximum(pred_test, pred_test_upper)

            mse_train = sklearn.metrics.mean_squared_error(data[:n_train], pred_train)
            mse_valid = sklearn.metrics.mean_squared_error(data[n_train:], pred_valid)
            print('mse_train', mse_train)
            print('mse_valid', mse_valid)
            print('mse', mse)

            self.content = content
            self.data = data
            self.data_all = data_all
            self.basename = basename
            self.n_train = n_train
            self.mse = mse
            self.pred_train = pred_train
            self.pred_train_lower = pred_train_lower
            self.pred_train_upper = pred_train_upper
            self.pred_valid = pred_valid
            self.pred_valid_lower = pred_valid_lower
            self.pred_valid_upper = pred_valid_upper
            self.pred_test = pred_test
            self.pred_test_lower = pred_test_lower
            self.pred_test_upper = pred_test_upper
            self.mse_train = mse_train
            self.mse_valid = mse_valid
            self.dates = dates
