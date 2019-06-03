if (NOT LITE_WITH_CL)
    return()
endif()

#include_directories("/usr/local/cuda-8.0/include")

cc_library(cl_helper_lite SRCS helper.cc)
cc_library(cl_context SRCS cl_context.cc)
cc_library(cl_allocator_lite SRCS allocator.cc)
