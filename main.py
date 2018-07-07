import os
# uncomment to force CPU training
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
import json
from config import Config
from rnn_predictor import RNNPredictor
from json_writer import JSONWriter
from plot_writer import PlotWriter

config = Config()
predictor = RNNPredictor(config)
json_w = JSONWriter(config)
plot_w = PlotWriter(config)

filename = sys.argv[1]
content = json.loads(open(filename).read())
model = predictor.predict(content)
json_w.write(model)
plot_w.write(model)