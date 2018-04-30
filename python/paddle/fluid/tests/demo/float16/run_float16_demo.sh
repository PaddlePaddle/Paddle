#!/bin/bash
mkdir -p /paddle/build && cd /paddle/build
cmake .. -DWITH_AVX=OFF -DWITH_MKL=OFF -DWITH_GPU=ON -DWITH_TESTING=ON -DWITH_TIMER=ON -DWITH_PROFILER=ON -DWITH_FLUID_ONLY=ON
make -j 32
pip install -U "/paddle/build/python/dist/$(ls /paddle/build/python/dist)"
cd /paddle/python/paddle/fluid/tests/demo/float16
python float16_inference_accuracy.py --threshold=0.6 --repeat=10 > float16_inference_accuracy.log
