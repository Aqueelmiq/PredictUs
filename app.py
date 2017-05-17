import os

from flask import Flask, request
from datetime import datetime
app = Flask(__name__)

import urllib.request

import tensorflow as tf

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def homepage():
    the_time = datetime.now().strftime("%A, %d %b %Y %l:%M %p")

    return """
    <h1>Hello heroku</h1>
    <p>It is currently {time}.</p>

    <img src="http://loremflickr.com/600/400" />
    """.format(time=the_time)

@app.route('/predict', methods=['POST'])
def predict():
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

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))

    return "HI"

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)

