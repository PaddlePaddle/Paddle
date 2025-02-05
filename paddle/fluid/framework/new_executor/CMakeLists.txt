file(GLOB_RECURSE standalone_executor_srcs "*.cc")

if(NOT (WITH_CINN))
  list(REMOVE_ITEM standalone_executor_srcs
       ${CMAKE_CURRENT_SOURCE_DIR}/instruction/cinn_jit_instruction.cc)
endif()

if(NOT WITH_ONEDNN)
  list(
    REMOVE_ITEM
    standalone_executor_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/instruction/onednn/onednn_legacy_instruction.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/instruction/onednn/onednn_instruction.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/instruction/onednn/onednn_mixed_instruction.cc)
endif()

if(NOT WITH_CUSTOM_DEVICE)
  list(REMOVE_ITEM standalone_executor_srcs
       ${CMAKE_CURRENT_SOURCE_DIR}/instruction/custom_engine_instruction.cc)
endif()

if(NOT TENSORRT_FOUND OR NOT WITH_TENSORRT)
  list(REMOVE_ITEM standalone_executor_srcs
       ${CMAKE_CURRENT_SOURCE_DIR}/instruction/tensorrt_engine_instruction.cc)
endif()

set(standalone_executor_deps
    pir
    program_translator
    op_dialect_vjp
    global_utils
    framework_io
    garbage_collector
    executor_gc_helper
    phi
    common)

if(WITH_CINN)
  set(standalone_executor_deps
      ${standalone_executor_deps}
      cinn_runtime_dialect
      phi
      common
      cinnapi
      device_context
      op_registry
      ${DEVICE_EVENT_LIBS})
endif()

if(TENSORRT_FOUND AND WITH_TENSORRT)
  set(standalone_executor_deps ${standalone_executor_deps} trt_engine)
endif()

cc_library(
  standalone_executor
  SRCS ${standalone_executor_srcs}
  DEPS ${standalone_executor_deps})

add_dependencies(standalone_executor xxhash)

if(WITH_ONEDNN)
  add_dependencies(standalone_executor onednn)
endif()
