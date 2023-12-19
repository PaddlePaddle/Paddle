file(GLOB_RECURSE new_executor_srcs "*.cc")

if(NOT (WITH_CINN AND NOT CINN_ONLY))
  list(REMOVE_ITEM new_executor_srcs
       ${CMAKE_CURRENT_SOURCE_DIR}/instruction/cinn_jit_instruction.cc)
endif()

set(new_executor_deps
    pir
    program_translator
    op_dialect_vjp
    pir_transforms
    garbage_collector
    executor_gc_helper
    framework_proto)

if(WITH_CINN AND NOT CINN_ONLY)
  set(new_executor_deps
      ${new_executor_deps}
      cinn_runtime_dialect
      buffered_reader
      device_context
      global_utils
      op_registry
      ${DEVICE_EVENT_LIBS})
endif()

cc_library(
  new_executor
  SRCS ${new_executor_srcs}
  DEPS ${new_executor_deps})
add_dependencies(new_executor xxhash)
if(WITH_MKLDNN)
  add_dependencies(new_executor mkldnn)
endif()
