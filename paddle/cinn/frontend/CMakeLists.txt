core_gather_headers()
gather_srcs(
  cinnapi_src
  SRCS
  computation.cc
  syntax.cc
  paddle_model_to_program.cc
  interpreter.cc
  net_builder.cc
  op_mapper_registry.cc
  paddle_model_convertor.cc
  program_pass.cc
  optimize.cc)

if(NOT WITH_CUDA)

  #  cinn_cc_test(test_frontend_interpreter
  #          ARGS --model_dir=${THIRD_PARTY_PATH}/naive_mul_model
  #          SRCS interpreter_test.cc DEPS cinncore)

else()
  cinn_nv_test(
    test_frontend_interpreter
    ARGS
    --model_dir=${THIRD_PARTY_PATH}/naive_mul_model
    SRCS
    interpreter_test.cc
    DEPS
    cinncore)
endif()

#cinn_cc_test(test_paddle_model_convertor
#        ARGS --model_dir=${THIRD_PARTY_PATH}/resnet_model
#        SRCS paddle_model_convertor_test.cc DEPS cinncore decomposer_test_helper)

#cinn_cc_test(test_computation
#  ARGS "--model_dir=${THIRD_PARTY_PATH}/naive_mul_model"
#  SRCS computation_test.cc DEPS cinncore)

add_subdirectory(paddle)
add_subdirectory(op_mappers)
# add_subdirectory(pass)
#add_subdirectory(group_cluster)
