if (NOT LITE_WITH_X86)
    return()
endif()

cc_library(target_wrapper_x86 SRCS target_wrapper.cc)
