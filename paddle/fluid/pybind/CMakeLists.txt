set(PYBIND_DEPS pybind python proto_desc memory executor fleet_wrapper box_wrapper nccl_wrapper prune
  feed_fetch_method pass_builder parallel_executor profiler layer tracer engine scope_pool
  analysis_predictor imperative_profiler nccl_context imperative_flag save_load_util dlpack_tensor)

if(WITH_PYTHON)
  list(APPEND PYBIND_DEPS py_func_op)
endif()

if (WITH_DISTRIBUTE)
  list(APPEND PYBIND_DEPS communicator)
endif()

set(PYBIND_SRCS
  pybind.cc
  exception.cc
  protobuf.cc
  const_value.cc
  reader_py.cc
  fleet_wrapper_py.cc
  box_helper_py.cc
  nccl_wrapper_py.cc
  data_set_py.cc
  imperative.cc
  ir.cc
  inference_api.cc)

if (WITH_DISTRIBUTE)
  list(APPEND PYBIND_SRCS communicator_py.cc)
endif()

if(WITH_PYTHON)
  list(APPEND PYBIND_DEPS ${GLOB_OP_LIB} ${GLOB_OPERATOR_DEPS})
  get_property(os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
  list(APPEND PYBIND_DEPS ${os_dependency_modules})

  add_executable(op_function_generator op_function_generator.cc)
  target_link_libraries(op_function_generator ${PYBIND_DEPS})

  add_custom_target(op_function_cmd 
    COMMAND "${CMAKE_BINARY_DIR}/paddle/fluid/pybind/op_function_generator" 
    "${CMAKE_SOURCE_DIR}/paddle/fluid/pybind/op_function_impl.h") 
  add_dependencies(op_function_cmd op_function_generator)

  if(WITH_AMD_GPU)
    hip_library(paddle_pybind SHARED SRCS ${PYBIND_SRCS} DEPS ARCHIVE_START ${PYBIND_DEPS} ARCHIVE_END)
  else()
    cc_library(paddle_pybind SHARED SRCS ${PYBIND_SRCS} DEPS ${PYBIND_DEPS})
    if(NOT APPLE AND NOT WIN32)
      target_link_libraries(paddle_pybind rt)
    endif()
  endif()

  add_dependencies(paddle_pybind op_function_cmd) 
endif(WITH_PYTHON)
