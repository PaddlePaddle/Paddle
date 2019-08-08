if (NOT LITE_WITH_X86)
    return()
endif()

lite_cc_library(target_wrapper_x86 SRCS target_wrapper.cc)
