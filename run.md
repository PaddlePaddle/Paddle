1. mkdir build & cd build
2. ../paddle/fluid/lite/tools/build.sh cmake_x86
3. make test_step_rnn_lite_x86 -j
4. ./paddle/fluid/lite/api/test_step_rnn_lite_x86 --model_dir=<model dir> --warmup=10000 --repeats=10000
