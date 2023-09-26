#!/bin/bash

model=$1

python3 convert-to-gguf.py ${model}/ 0
python3 convert-to-gguf.py ${model}/ 1
../build/bin/quantize ${model}/ggml-model-f16.gguf ${model}/ggml-model-q4_0.gguf 2
../build/bin/quantize ${model}/ggml-model-f16.gguf ${model}/ggml-model-q4_1.gguf 3