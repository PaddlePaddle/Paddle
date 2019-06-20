if(NOT (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK AND LITE_WITH_ARM))
    return()
endif()

if(NOT (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK AND LITE_WITH_ARM))
  return()
endif()

# TODO(xxx): seperate them
cc_library(math_arm SRCS  
    funcs.cc 
    packed_sgemm.cc 
    softmax.cc 
    scale.cc
    pooling.cc
    elementwise.cc
    concat.cc
    sgemv.cc
    type_trans.cpp
    conv_impl.cc
    conv_direct_3x3s1.cc
    conv_direct_3x3s2.cc
    conv_direct.cc
    conv_depthwise_3x3_int7.cc
    conv_depthwise_3x3_int8.cc
    conv_depthwise_5x5s1_int8.cc
    conv_depthwise_3x3p0.cc
    conv_depthwise_3x3p1.cc
    conv_depthwise_5x5s1.cc
    conv_depthwise_5x5s2.cc
    conv_depthwise.cc
    conv_gemmlike.cc
    conv_winograd_3x3.cc
    conv_winograd.cc
    split.cc
    activation.cc
    dropout.cc
    gemm_prepacked_int8.cc
    gemv_arm_int8.cc
    DEPS ${lite_kernel_deps} eigen3 framework_proto_lite)
  # TODO(TJ): fix me do not deps proto
 
 
