import os

# uncomment to force CPU training
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
import json
from config import Config
from rnn_predictor import RNNPredictor
from json_writer import JSONWriter
from plot_writer import PlotWriter

if __name__ == "__main__":
    config = Config()
    json_w = JSONWriter(config)
    plot_w = PlotWriter(config)
    predictor = RNNPredictor(config)

    filename = sys.argv[1]
    content = json.loads(open(filename).read())

    predictor.predict(content)
    plot_w.write(predictor)
    json_w.write(predictor)
