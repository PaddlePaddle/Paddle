list(APPEND SRCS generic_plugin.cu)

nv_library(
  dynamic_shape_infermeta
  SRCS dynamic_shape_infermeta.cc
  DEPS common pir)

nv_library(
  pir_tensorrt_plugin
  SRCS ${SRCS}
  DEPS common pir tensorrt_plugin dynamic_shape_infermeta trt_plugin
       pir_save_load)
