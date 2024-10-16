proto_library(framework_proto SRCS framework.proto)
proto_library(pass_desc_proto SRCS pass_desc.proto DEPS framework_proto)
proto_library(data_feed_proto SRCS data_feed.proto)
proto_library(trainer_desc_proto SRCS trainer_desc.proto DEPS framework_proto
              data_feed_proto)
proto_library(heter_service_proto SRCS heter_service.proto)

file(
  GLOB framework_cc
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "*.cc")
collect_srcs(core_srcs SRCS ${framework_cc})
