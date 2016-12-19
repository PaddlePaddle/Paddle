# Automatically init submodules
if(EXISTS ${PROJECT_SOURCE_DIR}/.gitmodules)
  # warp-ctc
  function(FindWarpCTC)
    set(WARPCTC_ROOT $ENV{WARPCTC_ROOT} CACHE PATH "Folder contains warp-ctc")
    find_path(WARPCTC_INCLUDE_DIR ctc.h PATHS
      ${WARPCTC_ROOT}/include ${PROJECT_SOURCE_DIR}/warp-ctc/include)
  endfunction(FindWarpCTC)

  FindWarpCTC()
  if(NOT WARPCTC_INCLUDE_DIR)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} submodule update --init -- warp-ctc
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      OUTPUT_VARIABLE GIT_OUTPUT
      RESULT_VARIABLE GIT_RESULT)
    message(STATUS ${GIT_OUTPUT})
    FindWarpCTC()
  endif()

  if(NOT WARPCTC_INCLUDE_DIR)
    message(FATAL_ERROR "warp-ctc must be set."
        "Try set WARPCTC_ROOT or run command git submodule --init -- warp-ctc.")
  else()
    message(STATUS "Found warp-ctc (include: ${WARPCTC_INCLUDE_DIR})")
  endif()
  include_directories(${WARPCTC_INCLUDE_DIR})
endif()
