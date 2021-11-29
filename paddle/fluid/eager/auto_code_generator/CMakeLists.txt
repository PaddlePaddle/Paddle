set(EAGER_GENERETOR_DEPS ${GLOB_OP_LIB} ${GLOB_OPERATOR_DEPS} pybind proto_desc executor layer tracer engine imperative_profiler imperative_flag)

add_executable(eager_generator eager_generator.cc)
target_link_libraries(eager_generator ${EAGER_GENERETOR_DEPS})

get_property (os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
target_link_libraries(eager_generator ${os_dependency_modules})

if(WITH_ROCM)
    target_link_libraries(eager_generator ${ROCM_HIPRTC_LIB})
endif()

# Prepare file structure
message("Generate dygraph file structure at path: ${PADDLE_SOURCE_DIR}/paddle/fluid/eager/generated")
execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" "${PADDLE_SOURCE_DIR}/paddle/fluid/eager/auto_code_generator/generate_file_structures.py" "${PADDLE_SOURCE_DIR}/paddle/fluid/eager/"
)

if(WIN32)
    add_custom_target(eager_codegen
      COMMAND "${CMAKE_CURRENT_BINARY_DIR}/eager_generator.exe" "${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/fluid_generated" 
      DEPENDS eager_generator
      VERBATIM)
else()
    add_custom_target(eager_codegen
          COMMAND "${CMAKE_CURRENT_BINARY_DIR}/eager_generator" "${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/fluid_generated"
          DEPENDS eager_generator
          VERBATIM)
endif()
