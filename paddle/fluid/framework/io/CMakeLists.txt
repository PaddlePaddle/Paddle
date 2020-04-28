cc_library(fs SRCS fs.cc DEPS string_helper glog boost)
cc_library(shell SRCS shell.cc DEPS string_helper glog timer enforce)

cc_test(test_fs SRCS test_fs.cc DEPS fs shell)
cc_test(fstream_ext_test SRCS fstream_ext_test.cc DEPS cryptopp)
