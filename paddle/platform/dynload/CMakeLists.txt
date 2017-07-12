cc_library(dynamic_loader SRCS dynamic_loader.cc DEPS glog gflags)
nv_library(dyload_cuda SRCS cublas.cc cudnn.cc curand.cc)
