# save current branch of third-party cutlass
execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}/third_party/cutlass
  OUTPUT_VARIABLE original_commit
  OUTPUT_STRIP_TRAILING_WHITESPACE)

# Switch to the specified commit eefa171, because the FP8 DeepGEMM requires the header files from this branch.
execute_process(COMMAND git checkout eefa171
                WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}/third_party/cutlass)

set(include_dirs
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/cutlass/include/cute"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/cutlass/include/cutlass")

set(build_include_dir "${CMAKE_CURRENT_SOURCE_DIR}/deep_gemm/include")
file(MAKE_DIRECTORY "${build_include_dir}")

# copy headers from third_party/cutlass
foreach(d IN LISTS include_dirs)
  get_filename_component(dirname ${d} NAME)
  set(dst_dir "${build_include_dir}/${dirname}")

  if(EXISTS "${dst_dir}")
    file(REMOVE_RECURSE "${dst_dir}")
  endif()
  file(COPY "${d}" DESTINATION "${build_include_dir}")
endforeach()

# switch to origin branch
execute_process(COMMAND git checkout ${original_commit}
                WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}/third_party/cutlass)
