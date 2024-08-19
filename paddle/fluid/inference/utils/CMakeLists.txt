cc_library(
  infer_io_utils
  SRCS io_utils.cc
  DEPS paddle_inference_api lod_tensor shape_range_info_proto)
cc_library(
  model_utils
  SRCS model_utils.cc
  DEPS proto_desc phi common)

cc_library(table_printer SRCS table_printer.cc)

proto_library(shape_range_info_proto SRCS shape_range_info.proto)
