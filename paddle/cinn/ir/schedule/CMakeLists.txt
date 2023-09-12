cinn_proto_library(schedule_desc_proto SRCS schedule_desc.proto)

core_gather_headers()
gather_srcs(cinnapi_src SRCS ir_schedule.cc ir_schedule_util.cc
            ir_schedule_error.cc schedule_desc.cc)

foreach(header ${schedule_desc_proto_HDRS})
  set(core_proto_includes
      "${core_proto_includes};${header}"
      CACHE INTERNAL "")
endforeach()
