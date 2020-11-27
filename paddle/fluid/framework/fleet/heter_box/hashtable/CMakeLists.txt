if(WITH_PSLIB)
    nv_library(gpu_ps SRCS gpu_ps.h  gpu_resource.cc
    gpu_resource.h  hashtable.h)
    nv_test(test_gpu_ps SRCS test_gpu_ps.cu feature_value.h DEPS gpu_ps cub)
endif(WITH_PSLIB)
