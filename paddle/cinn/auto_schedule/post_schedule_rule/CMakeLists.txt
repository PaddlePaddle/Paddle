core_gather_headers()

gather_srcs(cinnapi_src SRCS cooperative_process.cc)

if(WITH_CUDA)
  cinn_nv_test(
    test_cooperative_process
    SRCS
    cooperative_process_test.cc
    DEPS
    cinncore
    auto_gen_rule_test_helper
    test_program_builder)
endif()
