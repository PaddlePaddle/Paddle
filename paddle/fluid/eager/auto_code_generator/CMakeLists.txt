add_subdirectory(generator)

set(EAGER_GENERATOR_DEPS
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

if(WITH_CINN)
  list(REMOVE_ITEM EAGER_GENERATOR_DEPS imperative_flag)
endif()

if(WITH_CUSTOM_DEVICE)
  set(EAGER_GENERATOR_DEPS ${EAGER_GENERATOR_DEPS}
                           custom_device_common_op_registry)
endif()

get_property(os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)

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
