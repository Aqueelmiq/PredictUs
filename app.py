import os

from flask import Flask, request, jsonify
from datetime import datetime
from flask_cors import CORS
import urllib.request
import tensorflow as tf


app = Flask(__name__)
CORS(app)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def homepage():
    the_time = datetime.now().strftime("%A, %d %b %Y %l:%M %p")

    return "Hello"

@app.route('/predict', methods=['POST'])
def predict():

    print(request.json)
    url = request.json['img']
    image_path = os.path.join(APP_ROOT, 'tests.jpg')
    urllib.request.urlretrieve(url, image_path)


    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    labels = os.path.join(APP_ROOT, "retrained_labels.txt")

    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile(labels)]

    models = os.path.join(APP_ROOT, "retrained_graph.pb")

    with tf.gfile.FastGFile(models, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, \
                               {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        winner = label_lines[top_k[0]]

        diff12 = abs(predictions[0][0] - predictions[0][1])
        diff23 = abs(predictions[0][1] - predictions[0][2])
        diff34 = abs(predictions[0][2] - predictions[0][3])

        if(diff12 < 0.2):
            winner = 'confused'

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))

    os.remove(image_path)

    return jsonify({'winner': winner})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)

