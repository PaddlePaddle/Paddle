if(NOT WITH_ROCM)
  return()
endif()

# Now we don't support RCCL on windows
if(WIN32)
  return()
endif()

if(WITH_RCCL)
  set(RCCL_ROOT
      ${ROCM_PATH}/rccl
      CACHE PATH "RCCL ROOT")
  # ROCm 7.0+: rccl.h is under include/rccl/ directory
  # First try to find rccl.h directly (handles both old and new layouts)
  find_file(
    RCCL_HEADER_FILE rccl.h
    PATHS ${ROCM_PATH}/include/rccl
          ${ROCM_PATH}/include
          ${RCCL_ROOT}
          ${RCCL_ROOT}/include
          ${RCCL_ROOT}/local/include
          $ENV{RCCL_ROOT}
          $ENV{RCCL_ROOT}/include
          $ENV{RCCL_ROOT}/local/include
    NO_DEFAULT_PATH)

  if(NOT RCCL_HEADER_FILE)
    message(FATAL_ERROR "Cannot find rccl.h. Please check RCCL installation.")
  endif()

  # Get the directory containing rccl.h
  get_filename_component(RCCL_INCLUDE_DIR ${RCCL_HEADER_FILE} DIRECTORY)

  file(READ ${RCCL_HEADER_FILE} RCCL_VERSION_FILE_CONTENTS)

  string(REGEX MATCH "define NCCL_VERSION_CODE +([0-9]+)" RCCL_VERSION
               "${RCCL_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define NCCL_VERSION_CODE +([0-9]+)" "\\1" RCCL_VERSION
                       "${RCCL_VERSION}")

  # 2604 for ROCM3.5 and 2708 for ROCM 3.9
  message(STATUS "Current RCCL header is ${RCCL_HEADER_FILE}. "
                 "Current RCCL version is v${RCCL_VERSION}. ")
endif()
