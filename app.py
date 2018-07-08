import json
import subprocess

from flask import Flask, request, abort, send_from_directory

from config import Config

app = Flask(__name__)
config = Config()


@app.route('/', methods=['GET'])
def index():
    return app.send_static_file('index.html')


@app.route('/fig/<path:filename>')
def download_fig(filename):
    return send_from_directory('fig', filename)


@app.route('/model/<path:filename>')
def download_model(filename):
    return send_from_directory('model', filename)


@app.route('/output/<path:filename>')
def download_output(filename):
    return send_from_directory('output', filename)


@app.route('/log/<path:filename>')
def download_log(filename):
    return send_from_directory('log', filename)


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
        q_id = content['ifp']['id']
        filename = '{}andy_input_{}.json'.format(config.data_prefix, q_id)
        with open(filename, 'w') as fout:
            json.dump(content, fout)
        process = subprocess.Popen('python main.py ' + filename, shell=True)
        process.wait()
        if process.returncode != 0:
            abort(500)
        else:
            outputname = 'andy_output_{}.json'.format(q_id)
            return send_from_directory('output', outputname)


def main():
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)


if __name__ == "__main__":
    main()
