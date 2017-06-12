cc_library(stringpiece SRCS stringpiece.cc)
cc_test(stringpiece_test SRCS stringpiece_test.cc DEPS stringpiece glog gflags)
