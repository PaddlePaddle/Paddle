cc_library(fs SRCS fs.cc DEPS string_helper glog boost enforce)
cc_library(shell SRCS shell.cc DEPS string_helper glog timer enforce)

cc_test(test_fs SRCS test_fs.cc DEPS fs shell)
if (WITH_CRYPTO) 
    add_subdirectory(crypto)
endif (WITH_CRYPTO)
