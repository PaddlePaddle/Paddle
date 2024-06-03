cinn_proto_library(tile_config_proto SRCS tileconfig_desc.proto)

core_gather_headers()

gather_srcs(cinnapi_src SRCS group_tile_config.cc)
gather_srcs(cinnapi_src SRCS database.cc)

cinn_cc_library(file_tile_database SRCS filedatabase.cc DEPS absl)

foreach(header ${filetileconfig_proto_HDRS})
  set(core_proto_includes
      "${core_proto_includes};${header}"
      CACHE INTERNAL "")
endforeach()
