# Reproduction code for 43128

I provide the converted model which runs slowly on Kirin 990 and conversion script.
tf1/convert_tflite.py is a conversion script.
tf1/int8.tflite is a model which weights were converted as int8.
tf1/full_int8.tflite is a model which both input/output and weights were converted as int8.

Please check the difference between the model and mobilenet_v1_1.0_224_quant.tflite, which run fast on Kirin 990.
