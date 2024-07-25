list(APPEND SRCS generic_plugin.cu)

nv_library(
  dynamic_shape_infermeta
  SRCS dynamic_shape_infermeta.cc
  DEPS enforce pir)

nv_library(
  pir_tensorrt_plugin
  SRCS ${SRCS}
  DEPS enforce
       common
       pir
       tensorrt_plugin
       dynamic_shape_infermeta
       trt_plugin
       pir_save_load)
