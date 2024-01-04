if(XPU_SDK_ROOT)
  set(XPU_DEPS xpuapi xpurt)
endif()

cc_library(
  lite_op_teller
  SRCS op_teller.cc
  DEPS ${LITE_DEPS} framework_proto device_context xxhash)
cc_library(
  lite_engine
  SRCS engine.cc
  DEPS ${LITE_DEPS} framework_proto ${XPU_DEPS})
cc_library(
  lite_tensor_utils
  SRCS tensor_utils.cc
  DEPS ${LITE_DEPS} framework_proto device_context ${XPU_DEPS})
