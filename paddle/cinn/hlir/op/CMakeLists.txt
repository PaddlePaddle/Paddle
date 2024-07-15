add_subdirectory(contrib)

core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  nn.cc
  broadcast.cc
  transform.cc
  elementwise.cc
  reduction.cc
  op_util.cc
  custom_call.cc)
