
if(NOT (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK AND LITE_WITH_ARM))
  return()
endif()

cc_library(math_arm SRCS funcs.cc packed_sgemm.cc softmax.cc scale.cc elementwise.cc pooling.cc DEPS ${lite_kernel_deps} eigen3)
