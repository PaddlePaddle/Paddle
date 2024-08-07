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

if(NOT TENSORRT_FOUND OR NOT WITH_TENSORRT)
  list(REMOVE_ITEM standalone_executor_srcs
       ${CMAKE_CURRENT_SOURCE_DIR}/instruction/tensorrt_engine_instruction.cc)
endif()

set(standalone_executor_deps
    pir
    program_translator
    op_dialect_vjp
    global_utils
    buffered_reader
    framework_io
    garbage_collector
    executor_gc_helper
    device_event_base
    framework_proto)

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

if(WITH_CUSTOM_DEVICE)
  set(standalone_executor_deps ${standalone_executor_deps}
                               device_event_custom_device)
endif()

if(WITH_GPU)
  set(standalone_executor_deps ${standalone_executor_deps} device_event_gpu)
endif()

cc_library(
  standalone_executor
  SRCS ${standalone_executor_srcs}
  DEPS ${standalone_executor_deps})

add_dependencies(standalone_executor xxhash framework_proto)

if(WITH_ONEDNN)
  add_dependencies(standalone_executor onednn)
endif()
