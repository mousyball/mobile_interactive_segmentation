#!/bin/bash

TF_MODEL='tf.pb/'
TFLIFE_MODEL='tf.pb/test.tflite'

tflite_convert \
    --output_file=$TFLIFE_MODEL \
    --saved_model_dir=$TF_MODEL
    # --enable_v1_converter
    # --input_arrays=image, points \
    # --input_shapes=2,3,320,480:2,None,2 \
    # --output_arrays=output \
    # --output_shapes=2,1,320,480
