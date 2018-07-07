import os
import pdb

# uncomment to force CPU training
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import json
from flask import Flask, request, abort, jsonify, send_from_directory
from config import Config
from rnn_predictor import RNNPredictor
from json_writer import JSONWriter
from plot_writer import PlotWriter

app = Flask(__name__)
config = Config()
json_w = JSONWriter(config)
plot_w = PlotWriter(config)


@app.route('/', methods=['GET'])
def index():
    return app.send_static_file('index.html')


@app.route('/fig/<path:filename>')
def download_fig(filename):
    return send_from_directory('fig', filename)


@app.route('/model/<path:filename>')
def download_model(filename):
    return send_from_directory('model', filename)


@app.route('/api', methods=['POST'])
def api():
    content = None

    if request.is_json:
        content = request.json
    else:
        if 'json' in request.files:
            content = json.load(request.files.get('json'))

    if content is None:
        abort(400)
    else:
        predictor = RNNPredictor(config)
        model = predictor.predict(content)

        plot_w.write(model)
        res = json_w.write(model)
        predictor.close()
        return jsonify(res)


def main():
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
