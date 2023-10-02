cinn_proto_library(param_proto SRCS schedule_param.proto)

core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  broadcast.cc
  elementwise.cc
  nn.cc
  nn_util.cc
  reduction.cc
  load_x86_params.cc
  schedule.cc
  ir_schedule_pe.cc
  transform.cc
  vision.cc)

cinn_cc_test(test_cinn_pe_elementwise SRCS pe_elementwise_test.cc DEPS cinncore)
cinn_cc_test(test_cinn_pe_broadcast SRCS pe_broadcast_test.cc DEPS cinncore)
cinn_cc_test(test_cinn_pe_transform SRCS pe_transform_test.cc DEPS cinncore)
cinn_cc_test(test_load_params SRCS load_params_test.cc DEPS cinncore)

foreach(header ${param_proto_HDRS})
  set(core_proto_includes
      "${core_proto_includes};${header}"
      CACHE INTERNAL "")
endforeach()
