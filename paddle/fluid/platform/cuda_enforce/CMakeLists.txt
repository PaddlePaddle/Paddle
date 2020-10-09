if(WITH_GPU)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -dc")
    nv_library(cuda_enforce SRCS cuda_enforce.cu cuda_enforce.cuh DEPS enforce)
endif(WITH_GPU)
