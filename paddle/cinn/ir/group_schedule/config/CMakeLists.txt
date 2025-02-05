cinn_proto_library(tile_config_proto SRCS tile_config_desc.proto)

core_gather_headers()

gather_srcs(cinnapi_src SRCS group_tile_config.cc)
gather_srcs(cinnapi_src SRCS group_tile_util.cc)
gather_srcs(cinnapi_src SRCS database.cc)
gather_srcs(cinnapi_src SRCS file_database.cc)
gather_srcs(cinnapi_src SRCS schedule_config_manager.cc)

foreach(header ${file_tile_config_proto_HDRS})
  set(core_proto_includes
      "${core_proto_includes};${header}"
      CACHE INTERNAL "")
endforeach()
