if(NOT WITH_MUSA)
  return()
endif()

# Now we don't support MCCL on windows
if(WIN32)
  return()
endif()

# FIXME(MTAI): please make sure that we can find MCCL successfully
if(WITH_MCCL)
  set(MCCL_ROOT
      ${MUSA_PATH}/mccl
      CACHE PATH "MCCL ROOT")
  find_path(
    MCCL_INCLUDE_DIR mccl.h
    PATHS ${MCCL_ROOT} ${MCCL_ROOT}/include ${MCCL_ROOT}/local/include
          $ENV{MCCL_ROOT} $ENV{MCCL_ROOT}/include $ENV{MCCL_ROOT}/local/include
    NO_DEFAULT_PATH)

  file(READ ${MCCL_INCLUDE_DIR}/mccl.h MCCL_VERSION_FILE_CONTENTS)

  string(REGEX MATCH "define NCCL_VERSION_CODE +([0-9]+)" MCCL_VERSION
               "${MCCL_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define NCCL_VERSION_CODE +([0-9]+)" "\\1" MCCL_VERSION
                       "${MCCL_VERSION}")

  message(STATUS "Current MCCL header is ${MCCL_INCLUDE_DIR}/mccl.h. "
                 "Current MCCL version is v${MCCL_VERSION}. ")
endif()

