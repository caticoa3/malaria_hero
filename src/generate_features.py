from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
import tflite_runtime.interpreter as tflite
from tensorflow.python.platform import gfile
# import pandas as pd

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'



class FeatureGen(object):
    def __init__(self):
        #  self.image_data_tensor = JPEG_DATA_TENSOR_NAME
        #  self.bottleneck_tensor = BOTTLENECK_TENSOR_NAME
        self.load_graph()
        self.sess = tf.compat.v1.Session()

    def load_graph(self):
        self.model_filename = os.path.join(
            '../models','classify_image_graph_def.pb')

        with tf.compat.v1.Session() as sess:
            with gfile.GFile(self.model_filename, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                (self.bottleneck_tensor,
                 self.image_data_tensor,
                 self.resized_input_tensor) = tf.import_graph_def(
                     graph_def, name='', return_elements=[
                         BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                         RESIZED_INPUT_TENSOR_NAME])
                self.graph = sess.graph

    def load_tflit(self):
        # Load TFLite model and allocate tensors.
        interpreter = tflite.Interpreter(
            model_path="../models/quantized/trained/model.tflite")
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test model on random input data.
        input_shape = input_details[0]['shape']
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(output_data)


    def feature_gen(self, img_path):
        """
        high-level api to return bottleneck values
        """
        img_data = self.read_img_from_path(img_path)
        return self.run_tf_model_bottleneck(img_data)

    def read_img_from_path(self, img_path):
        return gfile.GFile(img_path, 'rb').read()

    def run_tf_model_bottleneck(self, img_data):
        bottleneck_values = self.sess.run(
            self.bottleneck_tensor,
            {self.image_data_tensor: img_data})
        return np.squeeze(bottleneck_values)

    def session_close(self):
        self.sess.close()

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(sys.argv)
        exit('usage: %prog <img_path>')

    gen = FeatureGen()
#    print(gen.feature_gen('../../Web.jpg'))
