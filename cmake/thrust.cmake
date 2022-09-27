function(add_thrust_patches_if_necessary)
  set(thrust_detect_file ${PROJECT_BINARY_DIR}/detect_thrust.cu)
  file(
    WRITE ${thrust_detect_file}
    ""
    "#include \"thrust/version.h\"\n"
    "#include \"thrust/shuffle.h\"\n"
    "#include \"stdio.h\"\n"
    "int main() {\n"
    "  int version = THRUST_VERSION;\n"
    "  printf(\"%d\", version);\n"
    "  return 0;\n"
    "}\n")

  execute_process(
    COMMAND "${CUDA_NVCC_EXECUTABLE}" "--run" "${thrust_detect_file}"
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
    RESULT_VARIABLE nvcc_res
    ERROR_QUIET)
  if(NOT nvcc_res EQUAL 0)
    set(thrust_patches "${PADDLE_SOURCE_DIR}/patches/thrust")
    message(STATUS "Add thrust patches: ${thrust_patches}")
    include_directories(${thrust_patches})
  endif()
endfunction()

add_thrust_patches_if_necessary()
