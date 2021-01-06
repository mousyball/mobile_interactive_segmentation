import tensorflow as tf
import os

TF_PATH = "./tf.pb/saved_model.pb" # where the forzen graph is stored
TFLITE_PATH = "./tf.pb/test.tflite"
# protopuf needs your virtual environment to be explictly exported in the path
os.environ["PATH"] = "/home/user/miniconda3/envs/fbrs/bin:/home/user/miniconda3/bin:/usr/local/sbin:...."

# make a converter object from the saved tensorflow file
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    TF_PATH,  # TensorFlow freezegraph .pb model file
    # name of input arrays as defined in torch.onnx.export function before.
    input_arrays=['image', 'points'],
    # name of output arrays defined in torch.onnx.export function before.
    output_arrays=['output']
)

# tell converter which type of optimization techniques to use
# to view the best option for optimization read documentation of tflite about optimization
# go to this link https://www.tensorflow.org/lite/guide/get_started#4_optimize_your_model_optional
# converter.optimizations = [tf.compat.v1.lite.Optimize.DEFAULT]

converter.experimental_new_converter = True

# I had to explicitly state the ops
converter.target_spec.supported_ops = [tf.compat.v1.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.compat.v1.lite.OpsSet.SELECT_TF_OPS]

tf_lite_model = converter.convert()
# Save the model.
with open(TFLITE_PATH, 'wb') as f:
    f.write(tf_lite_model)
