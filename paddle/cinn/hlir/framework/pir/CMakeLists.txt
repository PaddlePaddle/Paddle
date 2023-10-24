if(NOT CINN_ONLY)
  core_gather_headers()
  gather_srcs(cinnapi_src SRCS utils.cc op_lowering_impl.cc op_mapper.cc)
endif()
