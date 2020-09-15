import sys
import numpy as np
import tensorflow as tf
from board import Board, PASS

if len(sys.argv) < 3:
    print(f"Usage {sys.argv[0]} <saved model dir> <tflite file> [float16/int8/full_int8]")
    sys.exit()

converter = tf.lite.TFLiteConverter.from_saved_model(sys.argv[1])

if len(sys.argv) == 4:
    if sys.argv[3] == "float16":
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
        converter.target_spec.supported_types = [tf.float16]
    elif sys.argv[3] == "int8" or sys.argv[3] == "full_int8":
        converter.experimental_enable_mlir_converter = True
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
        def representative_dataset_gen():
            b = Board()
            yield [np.array(b.feature().reshape([1, 19, 19, 18]), dtype=np.float32)]
            while b.random_play() != PASS:
                yield [np.array(b.feature().reshape([1, 19, 19, 18]), dtype=np.float32)]

        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        if sys.argv[3] == "full_int8":
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

tflite_model = converter.convert()
open(sys.argv[2], "wb").write(tflite_model)
