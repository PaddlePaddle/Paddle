cc_library(stringpiece SRCS piece.cc)
cc_test(stringpiece_test SRCS piece_test.cc DEPS stringpiece glog gflags)
