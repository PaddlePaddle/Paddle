cc_library(cpu_info SRCS cpu_info.cc)
cc_test(cpu_info_test SRCS cpu_info_test.cc DEPS cpu_info gflags glog)

nv_library(gpu_info SRCS gpu_info.cc)

cc_library(place SRCS place.cc)
cc_test(place_test SRCS place_test.cc DEPS place glog gflags)
