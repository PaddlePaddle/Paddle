function(add_thrust_patches_if_necessary)
  # ROCm 7.0+ has rocThrust shuffle support built-in, so no patches needed.
  if(WITH_ROCM
     AND DEFINED PADDLE_ROCM_VERSION
     AND PADDLE_ROCM_VERSION GREATER_EQUAL 70000000)
    message(STATUS "ROCm 7.0+ detected, skipping thrust patches")
    return()
  endif()

  # ROCm < 7.0 still needs thrust patches.
  if(WITH_ROCM)
    set(thrust_patches "${PADDLE_SOURCE_DIR}/patches/thrust")
    message(STATUS "ROCm < 7.0 detected, add thrust patches: ${thrust_patches}")
    include_directories(${thrust_patches})
    return()
  endif()

  # For CUDA, check if thrust has shuffle support
  if(WITH_GPU)
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
  endif()
endfunction()

add_thrust_patches_if_necessary()
