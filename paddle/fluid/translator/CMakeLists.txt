set(PD_PROGRAM_TRANSLATOR_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(PD_PROGRAM_TRANSLATOR_BINARY_DIR
    "${PADDLE_BINARY_DIR}/paddle/fluid/translator")

file(GLOB PD_PROGRAM_TRANSLATOR_SRCS "*.cc")

cc_library(
  program_translator
  SRCS ${PD_PROGRAM_TRANSLATOR_SRCS}
  DEPS proto_desc pd_dialect new_ir framework_proto)
