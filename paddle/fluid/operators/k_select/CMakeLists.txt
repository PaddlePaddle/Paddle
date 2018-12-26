if(NOT WITH_GPU)
    return()
endif()

nv_library(k_select SRCS common.cu k_select.cu)
