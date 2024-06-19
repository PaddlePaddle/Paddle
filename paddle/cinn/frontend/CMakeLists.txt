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

add_subdirectory(paddle)
