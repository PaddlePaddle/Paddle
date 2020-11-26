if(WITH_PSLIB)
    nv_library(gpu_ps SRCS gpu_ps.cu  gpu_ps.h  gpu_resource.cc
    gpu_resource.h  hashtable.cu  hashtable.h)
endif(WITH_PSLIB)
