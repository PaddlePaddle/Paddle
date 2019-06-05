
if(NOT (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK AND LITE_WITH_ARM))
  return()
endif()


# TODO(xxx): seperate them
cc_library(math_arm SRCS  
    funcs.cc 
    packed_sgemm.cc 
    softmax.cc 
    scale.cc
    sgemv.cc
    conv_impl.cc
    conv_direct_3x3s1.cc
    conv_direct_3x3s2.cc
    conv_direct.cc
    DEPS ${lite_kernel_deps} eigen3)
 
