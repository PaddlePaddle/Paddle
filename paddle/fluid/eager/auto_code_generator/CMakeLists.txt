add_subdirectory(generator)

set(EAGER_GENERETOR_DEPS
    ${GLOB_OP_LIB}
    ${GLOB_OPERATOR_DEPS}
    pybind
    proto_desc
    executor
    layer
    tracer
    engine
    imperative_profiler
    imperative_flag)

add_executable(eager_generator eager_generator.cc)
target_link_libraries(eager_generator ${EAGER_GENERETOR_DEPS})

get_property(os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
target_link_libraries(eager_generator ${os_dependency_modules})

if(WITH_ROCM)
  target_link_libraries(eager_generator ${ROCM_HIPRTC_LIB})
endif()

# Prepare file structure
message(
  "Generate dygraph file structure at path: ${PADDLE_SOURCE_DIR}/paddle/fluid/eager/generated"
)

set(CODE_GEN_SPLIT_FILE_COUNT "8")

execute_process(
  COMMAND
    "${PYTHON_EXECUTABLE}"
    "${PADDLE_SOURCE_DIR}/paddle/fluid/eager/auto_code_generator/generate_file_structures.py"
    "${PADDLE_SOURCE_DIR}/paddle/fluid/eager/" "${CODE_GEN_SPLIT_FILE_COUNT}")

if(WIN32)
  set(EAGER_CODEGEN_DEPS eager_generator)
  if("${CMAKE_GENERATOR}" STREQUAL "Ninja")
    set(eager_generator_path "${CMAKE_CURRENT_BINARY_DIR}")
  else()
    set(eager_generator_path "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}")
  endif()

  if(${CBLAS_PROVIDER} STREQUAL MKLML)
    message("Copied libiomp5md.dll for Eager AutoCodeGen")
    add_custom_command(
      OUTPUT ${eager_generator_path}/libiomp5md.dll
      COMMAND ${CMAKE_COMMAND} -E copy ${MKLML_SHARED_IOMP_LIB}
              ${eager_generator_path}
      DEPENDS mklml)
    list(APPEND EAGER_CODEGEN_DEPS ${eager_generator_path}/libiomp5md.dll)
  else()
    message("Copied openblas.dll for Eager AutoCodeGen")
    add_custom_command(
      OUTPUT ${eager_generator_path}/openblas.dll
      COMMAND ${CMAKE_COMMAND} -E copy ${OPENBLAS_SHARED_LIB}
              ${eager_generator_path}
      DEPENDS extern_openblas)
    list(APPEND EAGER_CODEGEN_DEPS ${eager_generator_path}/openblas.dll)
  endif()

  if(WITH_MKLDNN)
    message("Copied mkldnn.dll for Eager AutoCodeGen")
    add_custom_command(
      OUTPUT ${eager_generator_path}/mkldnn.dll
      COMMAND ${CMAKE_COMMAND} -E copy ${MKLDNN_SHARED_LIB}
              ${eager_generator_path}
      DEPENDS mkldnn)
    list(APPEND EAGER_CODEGEN_DEPS ${eager_generator_path}/mkldnn.dll)
  endif()

  if(WITH_ONNXRUNTIME)
    message("Copied onnxruntime for Eager AutoCodeGen")
    add_custom_command(
      OUTPUT ${eager_generator_path}/onnxruntime.dll
      COMMAND ${CMAKE_COMMAND} -E copy ${ONNXRUNTIME_SHARED_LIB}
              ${eager_generator_path}
      DEPENDS onnxruntime)
    list(APPEND EAGER_CODEGEN_DEPS ${eager_generator_path}/onnxruntime.dll)
    add_custom_command(
      OUTPUT ${eager_generator_path}/paddle2onnx.dll
      COMMAND ${CMAKE_COMMAND} -E copy ${PADDLE2ONNX_LIB}
              ${eager_generator_path}
      DEPENDS paddle2onnx)
    list(APPEND EAGER_CODEGEN_DEPS ${eager_generator_path}/paddle2onnx.dll)
  endif()

  add_custom_target(
    legacy_eager_codegen
    COMMAND
      "${eager_generator_path}/eager_generator.exe"
      "${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/fluid_generated"
      "${CODE_GEN_SPLIT_FILE_COUNT}"
    DEPENDS ${EAGER_CODEGEN_DEPS}
    VERBATIM)
else()
  add_custom_target(
    legacy_eager_codegen
    COMMAND
      ${CMAKE_COMMAND} -E env
      "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${CMAKE_CURRENT_BINARY_DIR}/../../pybind"
      "${CMAKE_CURRENT_BINARY_DIR}/eager_generator"
      "${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/fluid_generated"
      "${CODE_GEN_SPLIT_FILE_COUNT}"
    DEPENDS eager_generator
    VERBATIM)
endif()
