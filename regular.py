import os

# uncomment to force CPU training
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
import pdb
import json
import keras
from keras import backend as K
import sklearn
import sklearn.preprocessing
import sklearn.linear_model
import argparse
import numpy as np
import scipy
import scipy.stats
import pandas as pd
import dateutil.parser
import datetime
from dateutil.relativedelta import relativedelta
import tensorflow as tf
import matplotlib.pyplot as plt

# uncomment to force CPU training
# config = tf.ConfigProto(device_count = {'GPU': 0})
config = tf.ConfigProto()
sess = tf.Session(config=config)
keras.backend.set_session(sess)

filename = sys.argv[1]
basename = os.path.basename(filename)
model_prefix = 'model/'
fig_prefix = 'fig/'
output_prefix = 'output/'

regular_config = {
    'n_batch': 1,
    'n_max_epoch': 1000,
    'n_neurons': 32,
    'n_timestep': 1,
    'n_predict_step': 10,
    'n_input_dim': 1,
    'n_output_dim': 10,
    'n_patience': 10,
    'n_lr_decay': 2,
    'lr_decay': 0.99,
    'ratio_valid': 0.1,
    'max_valid': 100,
    'valid_loss_weight': 0.5
}

content = json.loads(open(filename).read())
array = content['payload']['historical_data']['ts']
data_all = [t[1] for t in array]
data = data_all[:-regular_config['n_predict_step']]
dates = [dateutil.parser.parse(t[0]) for t in array]


def get_weighted_loss_upper():
    def weighted_loss(y_true, y_pred):
        diff = y_pred - y_true
        diff_p = K.mean(K.square(K.clip(diff, 0, 1e10)), axis=-1)
        diff_n = K.mean(K.square(K.clip(diff, -1e10, 0)), axis=-1)
        return diff_p * 0.05 + diff_n * 1.95

    return weighted_loss


def get_weighted_loss_lower():
    def weighted_loss(y_true, y_pred):
        diff = y_pred - y_true
        diff_p = K.mean(K.square(K.clip(diff, 0, 1e10)), axis=-1)
        diff_n = K.mean(K.square(K.clip(diff, -1e10, 0)), axis=-1)
        return diff_p * 1.95 + diff_n * 0.05

    return weighted_loss


def train_standard(data_scaled, config):
    # propagate config into local namespace
    df = pd.DataFrame(data_scaled)
    df_array = [df.shift(1), df]
    for i in range(-1, -config.n_predict_step, -1):
        df_array.append(df.shift(i))
    df = pd.concat(df_array, axis=1)
    df.fillna(value=0, inplace=True)
    x, y = df.values[:, 0], df.values[:, 1:]
    x = x.reshape(-1, config.n_timestep, config.n_input_dim)
    y = np.repeat(y, 3, axis=0).reshape(len(x), config.n_output_dim * 3)

    n_valid = max(min(int(len(y) * config.ratio_valid), config.max_valid), 1)
    n_train = int(int((len(y) - n_valid) / config.n_batch) * config.n_batch)
    x_train, y_train = x[:n_train, :, :], y[:n_train]
    x_valid, y_valid = x[n_train:, :, :], y[n_train:]

    if config.n_predict_step != 1:
        y_train = [a.reshape(-1, 1, 1) for a in np.hsplit(y_train, config.n_predict_step * 3)]
        y_valid = [a.reshape(-1, 1, 1) for a in np.hsplit(y_valid, config.n_predict_step * 3)]
    else:
        y_train = y_train.reshape(-1, 1, 1)
        y_valid = y_valid.reshape(-1, 1, 1)

    # build network
    inputs = keras.layers.Input(shape=(config.n_timestep, config.n_input_dim),
                                batch_shape=(config.n_batch, config.n_timestep, config.n_input_dim))
    grus = []
    denses = []

    res_grus = []
    outputs = []

    for z in range(config.n_predict_step):
        gru = keras.layers.GRU(config.n_neurons, stateful=True, return_sequences=True, return_state=False)
        grus.append(gru)

        for y in range(3):
            dense = keras.layers.Dense(1)
            denses.append(dense)

    for z in range(config.n_predict_step):
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
    # weights = np.linspace(1, 0, num=n_predict_step+1)[:-1].tolist()
    # weights = [i+1 for i in range(config.n_predict_step)]
    # weights.reverse()
    mse_loss = ['mean_squared_error'] * 10
    upper_loss = [get_weighted_loss_upper() for zz in range(10)]
    lower_loss = [get_weighted_loss_lower() for zz in range(10)]
    losses = mse_loss + upper_loss + lower_loss
    model.compile(loss=losses, optimizer=adam)
    model.summary()

    # make valid model
    smallest_loss = float('inf')
    smallest_weight = None
    wait = 0

    # fit network
    for i in range(config.n_max_epoch):
        print('epoch: {}/{}'.format(i, config.n_max_epoch))
        # because
        loss = model.fit(x_train, y_train, epochs=1, batch_size=config.n_batch, verbose=1, shuffle=False,
                         validation_data=(x_valid, y_valid))
        model.reset_states()

        print('train loss: {}, valid loss: {}'.format(loss.history['loss'][0], loss.history['val_loss'][0]))
        total_loss = loss.history['loss'][0] * (1 - config.valid_loss_weight) + loss.history['val_loss'][
            0] * config.valid_loss_weight
        # push into heap
        if wait <= config.n_patience:
            if total_loss < smallest_loss:
                smallest_loss = total_loss
                smallest_weight = model.get_weights()
                wait = 0
                print('New smallest')
            else:
                wait += 1
                print('Wait {}'.format(wait))
                if wait % config.n_lr_decay == 0:
                    this_lr = adam.lr.eval(sess)
                    assign_op = adam.lr.assign(this_lr * config.lr_decay)
                    sess.run(assign_op)
                    print('Reduce lr to: ', this_lr * config.lr_decay)
        else:
            break

    model.set_weights(smallest_weight)
    return smallest_loss, model, n_train


def calc_prob_option(pred, pred_upper, pred_lower):
    if content['ifp']['ifp']['parsed_answers']['unit'] == 'boolean':
        value_list = [[None, 0.5], [0.5, None]]
    else:
        option_text = content['ifp']['ifp']['parsed_answers']['values']
        count = len(option_text)
        value_list = []

        for text in option_text:
            if text[0] == '<':
                values = [None, float(text[1:])]
            elif text[0] == '>':
                values = [float(text[1:]), None]
            else:
                values = [float(t) for t in text.split(' - ')]

            value_list.append(values)

    rv_upper = scipy.stats.norm(loc=pred, scale=(pred_upper - pred) / 2)
    rv_lower = scipy.stats.norm(loc=pred, scale=(pred - pred_lower) / 2)

    def return_prob(value):
        if value <= pred:
            return rv_lower.cdf(value)
        else:
            return rv_upper.cdf(value)

    prob_list = []
    for value in value_list:
        low, high = value
        if low == None:
            prob_list.append(return_prob(high))
        elif high == None:
            prob_list.append(1 - return_prob(low))
        else:
            prob_list.append(return_prob(high) - return_prob(low))

    prob_list = np.asarray(prob_list)
    print(sum(prob_list))
    prob_list = prob_list / sum(prob_list)
    print(prob_list)
    return prob_list


# prepare data
min_max_scaler = sklearn.preprocessing.StandardScaler()
data_scaled = min_max_scaler.fit_transform(np.asarray(data).reshape(-1, 1))

print('Run regular')
loss, model, n_train = train_standard(data_scaled, argparse.Namespace(**regular_config))
config = argparse.Namespace(**regular_config)
model.save(model_prefix + basename.replace('json', 'h5'))

data_forcast = np.insert(data_scaled, 0, 0)

# forecast
model.reset_states()
pred = model.predict(data_forcast.reshape(-1, 1, 1), batch_size=config.n_batch)
pred = np.asarray(pred).squeeze().T

if config.n_predict_step == 1:
    pred = pred.reshape(-1, 1)

mse = sklearn.metrics.mean_squared_error(data_scaled, pred[:-1, 0])

forecast_is_usable = 1
if mse > 0.1:
    print('mse is too large')
    forecast_is_usable = 0

pred = min_max_scaler.inverse_transform(pred)
pred_train = pred[:n_train, 0]
pred_valid = pred[n_train:-1, 0]
pred_train_upper = pred[:n_train, 10]
pred_train_lower = pred[:n_train, 20]
pred_valid_upper = pred[n_train:-1, 10]
pred_valid_lower = pred[n_train:-1, 20]
pred_test = pred[-1, :10]
pred_test_upper = pred[-1, 10:20]
pred_test_lower = pred[-1, 20:]

pred_train_upper = np.maximum(pred_train, pred_train_upper)
pred_train_lower = np.minimum(pred_train, pred_train_lower)
pred_valid_upper = np.maximum(pred_valid, pred_valid_upper)
pred_valid_lower = np.minimum(pred_valid, pred_valid_lower)
pred_test_upper = np.maximum(pred_test, pred_test_upper)
pred_test_lower = np.minimum(pred_test, pred_test_lower)

prob_list = calc_prob_option(pred_test[0], pred_test_upper[0], pred_test_lower[0])

mse_train = sklearn.metrics.mean_squared_error(data[:n_train], pred_train)
mse_valid = sklearn.metrics.mean_squared_error(data[n_train:], pred_valid)
print('mse_train', mse_train)
print('mse_valid', mse_valid)
print('mse', mse, loss)

plt.plot(list(range(len(data_all))), data_all, 'b', label='true')
plt.plot(list(range(n_train)), pred_train, 'g', label='pred_train')
plt.plot(list(range(n_train)), pred_train_upper, 'lime', label='pred_train_upper')
plt.plot(list(range(n_train)), pred_train_lower, 'darkgreen', label='pred_train_lower')
plt.plot(list(range(n_train, len(data))), pred_valid, 'r', label='pred_valid')
plt.plot(list(range(n_train, len(data))), pred_valid_upper, 'salmon', label='pred_valid_upper')
plt.plot(list(range(n_train, len(data))), pred_valid_lower, 'darkred', label='pred_valid_lower')
plt.plot(list(range(len(data), len(data) + len(pred_test))), pred_test, 'c', label='pred_test')
plt.plot(list(range(len(data), len(data) + len(pred_test))), pred_test_upper, 'aquamarine', label='pred_test_upper')
plt.plot(list(range(len(data), len(data) + len(pred_test))), pred_test_lower, 'darkcyan', label='pred_test_lower')
plt.plot([], [], ' ', label='mse_train: {0:.2f}'.format(mse_train))
plt.plot([], [], ' ', label='mse_valid: {0:.2f}'.format(mse_valid))
plt.plot([], [], ' ', label='mse: {}'.format(mse))
plt.legend()

plt.title('Forecast for ' + basename)
plt.savefig(fig_prefix + basename.replace('json', 'pdf'))
plt.close()

pred = np.insert(pred_test, 0, data[-1])
pred_upper = np.insert(pred_test_upper, 0, data[-1])
pred_lower = np.insert(pred_test_lower, 0, data[-1])
outputname = basename.replace('_input_', '_output_')

last_date = dates[-1]
diff = dates[-1] - dates[-2]
if diff.days in (365, 366):
    diff = relativedelta(years=1)
elif diff.days in (28, 29, 30, 31):
    diff = relativedelta(months=1)

pred = pred.tolist()
new_dates = [last_date + diff * i for i in range(len(pred))]
new_dates_str = [d.strftime("%Y-%m-%d") for d in new_dates]

res = {
    'forecast_is_usable': [forecast_is_usable],
    'forecasts': {
        'RNN': {
            'forecast_is_usable': [forecast_is_usable],
            'internal': {
                'rmse': [mse]
            },
            'model': ['RNN'],
            'to_date': [new_dates_str[-1]],
            'ts': [[new_dates_str[i], str(pred[i]), str(pred_lower[i]), str(pred_upper[i])] for i in range(len(pred))],
            'ts_colnames': [
                'date',
                'Point Forecast',
                'Lo 95',
                'Hi 95'
            ],
            'option_labels': content['ifp']['ifp']['parsed_answers']['values'],
            'option_probabilities': prob_list.tolist()
        }
    },
    'internal': {
        'rmse': [mse]
    },
    'model': ['RNN'],
    'option_labels': content['ifp']['ifp']['parsed_answers']['values'],
    'option_probabilities': prob_list.tolist(),
    'parsed_request': {
        'fcast_dates': new_dates_str[1:],
        'h': [config.n_predict_step],
        'target': {
            'date': [d.strftime("%Y-%m-%d") for d in dates],
            'value': data_all
        },
        'target_tail': {
            'date': [dates[-1].strftime("%Y-%m-%d")],
            'value': [data_all[-1]]
        }
    },
    'to_date': [new_dates_str[-1]],
    'ts': [[new_dates_str[i], str(pred[i]), str(pred_lower[i]), str(pred_upper[i])] for i in range(len(pred))],
    'ts_colnames': [
        'date',
        'Point Forecast',
        'Lo 95',
        'Hi 95'
    ],
}

with open(output_prefix + outputname, 'w') as fout:
    json.dump(res, fout)
