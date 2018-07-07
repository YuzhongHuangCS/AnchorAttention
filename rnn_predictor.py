import dateutil.parser
import keras
import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model
import sklearn.preprocessing
import tensorflow as tf
from keras import backend as K
import pdb

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
        sess = tf.Session()
        keras.backend.set_session(sess)

        array = content['payload']['historical_data']['ts']
        data_all = [t[1] for t in array]

        if self.config.test_split:
            data = data_all[:-self.config.n_predict_step]
        else:
            data = data_all

        basename = 'andy_input_{}.json'.format(content['ifp']['id'])
        dates = [dateutil.parser.parse(t[0]) for t in array]

        scaler = sklearn.preprocessing.StandardScaler()
        data_scaled = scaler.fit_transform(np.asarray(data).reshape(-1, 1))

        # propagate config into local namespace
        df = pd.DataFrame(data_scaled)
        df_array = [df.shift(1), df]
        for i in range(-1, -self.config.n_predict_step, -1):
            df_array.append(df.shift(i))
        df = pd.concat(df_array, axis=1)
        df.fillna(value=0, inplace=True)
        x, y = df.values[:, 0], df.values[:, 1:]
        x = x.reshape(-1, self.config.n_timestep, self.config.n_input_dim)
        y = np.repeat(y, 3, axis=0).reshape(len(x), self.config.n_output_dim * 3)

        n_valid = max(min(int(len(y) * self.config.ratio_valid), self.config.max_valid), 1)
        n_train = int(int((len(y) - n_valid) / self.config.n_batch) * self.config.n_batch)
        x_train, y_train = x[:n_train, :, :], y[:n_train]
        x_valid, y_valid = x[n_train:, :, :], y[n_train:]

        if self.config.n_predict_step != 1:
            y_train = [a.reshape(-1, 1, 1) for a in np.hsplit(y_train, self.config.n_predict_step * 3)]
            y_valid = [a.reshape(-1, 1, 1) for a in np.hsplit(y_valid, self.config.n_predict_step * 3)]
        else:
            y_train = y_train.reshape(-1, 1, 1)
            y_valid = y_valid.reshape(-1, 1, 1)

        # build network
        inputs = keras.layers.Input(shape=(self.config.n_timestep, self.config.n_input_dim),
                                    batch_shape=(self.config.n_batch, self.config.n_timestep, self.config.n_input_dim))
        grus = []
        denses = []

        res_grus = []
        outputs = []

        for z in range(self.config.n_predict_step):
            gru = keras.layers.GRU(self.config.n_neurons, stateful=True, return_sequences=True, return_state=False)
            grus.append(gru)

            for y in range(3):
                dense = keras.layers.Dense(1)
                denses.append(dense)

        for z in range(self.config.n_predict_step):
            if z == 0:
                res_gru = grus[z](inputs)
            else:
                res_gru = grus[z](res_grus[z - 1])

            res_grus.append(res_gru)

            for y in range(3):
                output = denses[z * 3 + y](res_gru)
                outputs.append(output)

        model = keras.models.Model(inputs=inputs, outputs=outputs)
        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        weights_mse = [i + 1 for i in range(self.config.n_predict_step)]
        weights_mse.reverse()

        weights_lower = [(i + 1) / 2 for i in range(self.config.n_predict_step)]
        weights_lower.reverse()

        weights_upper = [(i + 1) / 2 for i in range(self.config.n_predict_step)]
        weights_upper.reverse()

        weights = weights_mse + weights_lower + weights_upper
        weights.reverse()

        def get_weighted_loss_lower():
            def weighted_loss(y_true, y_pred):
                diff = y_pred - y_true
                diff_p = K.mean(K.square(K.clip(diff, 0, 1e10)), axis=-1)
                diff_n = K.mean(K.square(K.clip(diff, -1e10, 0)), axis=-1)
                return diff_p * 1.95 + diff_n * 0.05

            return weighted_loss

        def get_weighted_loss_upper():
            def weighted_loss(y_true, y_pred):
                diff = y_pred - y_true
                diff_p = K.mean(K.square(K.clip(diff, 0, 1e10)), axis=-1)
                diff_n = K.mean(K.square(K.clip(diff, -1e10, 0)), axis=-1)
                return diff_p * 0.05 + diff_n * 1.95

            return weighted_loss

        mse_loss = ['mean_squared_error'] * 10
        lower_loss = [get_weighted_loss_lower() for _ in range(10)]
        upper_loss = [get_weighted_loss_upper() for _ in range(10)]

        losses = mse_loss + lower_loss + upper_loss
        model.compile(loss=losses, optimizer=adam, loss_weights=weights)
        model.summary()

        # make valid model
        smallest_loss = float('inf')
        smallest_weight = None
        wait = 0

        # fit network
        for i in range(self.config.n_max_epoch):
            print('epoch: {}/{}'.format(i, self.config.n_max_epoch))
            # because
            loss = model.fit(x_train, y_train, epochs=1, batch_size=self.config.n_batch, verbose=1, shuffle=False,
                             validation_data=(x_valid, y_valid))
            model.reset_states()

            print('train loss: {}, valid loss: {}'.format(loss.history['loss'][0], loss.history['val_loss'][0]))
            total_loss = loss.history['loss'][0] * (1 - self.config.valid_loss_weight) + loss.history['val_loss'][
                0] * self.config.valid_loss_weight
            # push into heap
            if wait <= self.config.n_patience:
                if total_loss < smallest_loss:
                    smallest_loss = total_loss
                    smallest_weight = model.get_weights()
                    wait = 0
                    print('New smallest')
                else:
                    wait += 1
                    print('Wait {}'.format(wait))
                    if wait % self.config.n_lr_decay == 0:
                        this_lr = adam.lr.eval(sess)
                        assign_op = adam.lr.assign(this_lr * self.config.lr_decay)
                        sess.run(assign_op)
                        print('Reduce lr to: ', this_lr * self.config.lr_decay)
            else:
                break

        model.set_weights(smallest_weight)

        model.save(self.config.model_prefix + basename.replace('json', 'h5'))

        data_forcast = np.insert(data_scaled, 0, 0)
        model.reset_states()
        pred = model.predict(data_forcast.reshape(-1, 1, 1), batch_size=self.config.n_batch)
        pred = np.asarray(pred).squeeze().T

        if self.config.n_predict_step == 1:
            pred = pred.reshape(-1, 1)

        mse = sklearn.metrics.mean_squared_error(data_scaled, pred[:-1, 0])

        pred = scaler.inverse_transform(pred)
        pred_train = pred[:n_train, 0]
        pred_valid = pred[n_train:-1, 0]
        pred_train_lower = pred[:n_train, 10]
        pred_train_upper = pred[:n_train, 20]
        pred_valid_lower = pred[n_train:-1, 10]
        pred_valid_upper = pred[n_train:-1, 20]

        pred_test = pred[-1, :10]
        pred_test_lower = pred[-1, 10:20]
        pred_test_upper = pred[-1, 20:]

        pred_train_lower = np.minimum(pred_train, pred_train_lower)
        pred_train_upper = np.maximum(pred_train, pred_train_upper)
        pred_valid_lower = np.minimum(pred_valid, pred_valid_lower)
        pred_valid_upper = np.maximum(pred_valid, pred_valid_upper)
        pred_test_lower = np.minimum(pred_test, pred_test_lower)
        pred_test_upper = np.maximum(pred_test, pred_test_upper)

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
