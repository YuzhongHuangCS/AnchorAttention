import os
import sys
import pdb
import json
import sklearn
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math

# uncomment to force CPU training
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

filename = sys.argv[1]
basename = os.path.basename(filename)
model_prefix = 'model/'
fig_prefix = 'fig/'

content = json.loads(open(filename).read())
array = content['payload']['historical_data']['ts']
data = [t[1] for t in array]
min_max_scaler = sklearn.preprocessing.StandardScaler()
data_scaled = min_max_scaler.fit_transform(np.asarray(data).reshape(-1, 1))
data = data_scaled.squeeze()

num_epochs = 10
total_series_length = len(data)
truncated_backprop_length = 1
state_size = 32
num_classes = 1
predict_step = 1
batch_size = 1
num_batches = total_series_length//batch_size//truncated_backprop_length

#pdb.set_trace()
y = np.asarray(data)
x = np.roll(y, predict_step)
x[0:predict_step] = 0

x = x.reshape(1, -1)
y = y.reshape(1, -1)

batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])

cell_state = tf.placeholder(tf.float32, [batch_size, state_size])

limit = math.sqrt(6/(state_size + num_classes))
W2 = tf.Variable(np.random.uniform(-limit, limit, (state_size, num_classes)).astype(np.float32))
b2 = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float32)

# Unpack columns
inputs_series = tf.split(batchX_placeholder, truncated_backprop_length, 1)
labels_series = tf.unstack(batchY_placeholder, axis=1)

# Forward passes
cell = tf.nn.rnn_cell.GRUCell(state_size)
states_series, current_state = tf.nn.static_rnn(cell, inputs_series, cell_state)
prediction = tf.matmul(current_state, W2) + b2

total_loss = tf.squared_difference(prediction, labels_series)
train_step = tf.train.AdamOptimizer(0.001).minimize(total_loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	loss_list = []

	# train
	for epoch_idx in range(num_epochs):
		_current_cell_state = np.zeros((batch_size, state_size))
		loss_list = []
		for batch_idx in range(num_batches):
			start_idx = batch_idx * truncated_backprop_length
			end_idx = start_idx + truncated_backprop_length

			batchX = x[:, start_idx:end_idx]
			batchY = y[:, start_idx:end_idx]

			_total_loss, _train_step, _current_state = sess.run(
				[total_loss, train_step, current_state],
				feed_dict={
					batchX_placeholder: batchX,
					batchY_placeholder: batchY,
					cell_state: _current_cell_state,
				})

			_current_cell_state = _current_state

			loss_list.append(_total_loss)

		print("Epoch",epoch_idx, "Epoch loss", np.mean(loss_list))

	# test
	_current_cell_state = np.zeros((batch_size, state_size))
	preds = []
	for batch_idx in range(num_batches):
		start_idx = batch_idx * truncated_backprop_length
		end_idx = start_idx + truncated_backprop_length

		batchX = x[:, start_idx:end_idx]
		batchY = y[:, start_idx:end_idx]

		_prediction, _current_state = sess.run(
			[prediction, current_state],
			feed_dict={
				batchX_placeholder: batchX,
				batchY_placeholder: batchY,
				cell_state: _current_cell_state,
			})
		_current_cell_state = _current_state

		preds.append(_prediction)

	preds = np.asarray(preds).squeeze()
	y = y.squeeze()

	pred_inv = min_max_scaler.inverse_transform(np.asarray(preds).reshape(-1, 1)).squeeze()
	y_inv = min_max_scaler.inverse_transform(np.asarray(y).reshape(-1, 1)).squeeze()

	mse_train = sklearn.metrics.mean_squared_error(preds, y)

	plt.figure()
	plt.plot(list(range(len(preds))), y_inv, 'b', label='true')
	plt.plot(list(range(len(preds))), pred_inv, 'g', label='pred_train')
	plt.plot([], [], ' ', label='mse_train: {0:.2f}'.format(mse_train))
	plt.legend()
	plt.savefig(fig_prefix + basename.replace('json', 'pdf'))
	#plt.show()
