set(INTERPRETER_SRCS
    data_transfer.cc
    dependency_builder.cc
    execution_config.cc
    interpreter_util.cc
    static_build.cc
    stream_analyzer.cc
    plan.cc)

set(INTERPRETER_DEPS buffered_reader device_context global_utils op_registry
                     ${DEVICE_EVENT_LIBS})

if(WITH_CINN AND NOT CINN_ONLY)
  set(INTERPRETER_DEPS ${INTERPRETER_DEPS} cinn_jit_instruction)
endif()

cc_library(
  interpreter
  SRCS ${INTERPRETER_SRCS}
  DEPS standalone_executor ${INTERPRETER_DEPS})
