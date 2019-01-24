if(NOT WITH_GPU)
    return()
endif()

nv_library(encoded_all_reduce SRCS sparse.cu sparse_reduce.cu)
