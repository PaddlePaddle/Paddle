# Inference Model UT

There are several model tests currently:
- test_ernie_text_cls.cc
- test_LeViT.cc
- test_ppyolo_mbv3.cc
- test_ppyolov2_r50vd.cc
- test_resnet50.cc
- test_resnet50_quant.cc
- test_yolov3.cc

<<<<<<< HEAD
To build and execute tests on Linux, simply run
=======
To build and execute tests on Linux, simply run 
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
```
./run.sh $PADDLE_ROOT $TURN_ON_MKL $TEST_GPU_CPU $DATA_DIR
```
To build on windows, run command with busybox
```
busybox bash ./run.sh $PADDLE_ROOT $TURN_ON_MKL $TEST_GPU_CPU $DATA_DIR
```

- After run command, it will build and execute tests and download to ${DATA_DIR} automatically.
- `$PADDLE_ROOT`: paddle library path
- `$TURN_ON_MKL`: use MKL or Openblas
- `$TEST_GPU_CPU`: test both GPU/CPU mode or only CPU mode
- `$DATA_DIR`: download data path

<<<<<<< HEAD
now only support 4 kinds of tests which controled by `--gtest_filter` argument, test suite name should be same as following.
=======
now only support 4 kinds of tests which controled by `--gtest_filter` argument, test suite name should be same as following. 
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
- `TEST(gpu_tester_*, test_name)`
- `TEST(cpu_tester_*, test_name)`
- `TEST(mkldnn_tester_*, test_name)`
- `TEST(tensorrt_tester_*, test_name)`

skpied test suite name.
- `TEST(DISABLED_gpu_tester_*, test_name)`
- `TEST(DISABLED_cpu_tester_*, test_name)`
- `TEST(DISABLED_mkldnn_tester_*, test_name)`
- `TEST(DISABLED_tensorrt_tester_*, test_name)`
