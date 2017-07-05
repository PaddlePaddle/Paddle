cc_library(cpu_info SRCS cpu_info.cc DEPS gflags glog)
cc_test(cpu_info_test SRCS cpu_info_test.cc DEPS cpu_info)

nv_library(gpu_info SRCS gpu_info.cc DEPS gflags)

cc_library(place SRCS place.cc)
cc_test(place_test SRCS place_test.cc DEPS place glog gflags)

cc_library(dynamic_loader SRCS dynload/dynamic_loader.cc DEPS gflags)
