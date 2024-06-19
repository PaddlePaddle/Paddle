core_gather_headers()
gather_srcs(
  cinnapi_src
  SRCS
  syntax.cc
  paddle_model_to_program.cc
  net_builder.cc
  op_mapper_registry.cc
  paddle_model_convertor.cc
  program_pass.cc)

#cinn_cc_test(test_paddle_model_convertor
#        ARGS --model_dir=${THIRD_PARTY_PATH}/resnet_model
#        SRCS paddle_model_convertor_test.cc DEPS cinncore decomposer_test_helper)

#cinn_cc_test(test_computation
#  ARGS "--model_dir=${THIRD_PARTY_PATH}/naive_mul_model"
#  SRCS computation_test.cc DEPS cinncore)

add_subdirectory(paddle)
