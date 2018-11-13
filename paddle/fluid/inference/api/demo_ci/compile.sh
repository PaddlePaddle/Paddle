PADDLE_LIB=/home/chunwei/project/Paddle/build/fluid_inference_install_dir
g++ vis_demo.cc \
$PADDLE_LIB/third_party/install/glog/lib/libglog.a \
$PADDLE_LIB/third_party/install/gflags/lib/libgflags.a \
-I $PADDLE_LIB/include -lpaddle_fluid -lmklml_intel -o main \
-I $PADDLE_LIB/paddle/include \
-I $PADDLE_LIB/third_party/install/glog/include \
-I $PADDLE_LIB/third_party/install/gflags/include \
-L $PADDLE_LIB/third_party/install/mklml/lib \
-L $PADDLE_LIB/paddle/lib \
-lpthread \
-std=c++11

