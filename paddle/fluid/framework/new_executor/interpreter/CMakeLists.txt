cc_library(
  dependency_builder
  SRCS dependency_builder.cc
  DEPS operator)

cc_library(
  execution_config
  SRCS execution_config.cc
  DEPS phi_backends)
