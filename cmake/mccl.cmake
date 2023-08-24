if(NOT WITH_MUSA)
  return()
endif()

# Now we don't support MCCL on windows
if(WIN32)
  return()
endif()

if(WITH_MCCL)
  set(MCCL_ROOT
      "/usr/local/musa/"
      CACHE PATH "MCCL ROOT")
  find_path(
    MCCL_INCLUDE_DIR mccl.h
    PATHS ${MCCL_ROOT} ${MCCL_ROOT}/include ${MCCL_ROOT}/local/include
          $ENV{MCCL_ROOT} $ENV{MCCL_ROOT}/include $ENV{MCCL_ROOT}/local/include
    NO_DEFAULT_PATH)

  if(MCCL_INCLUDE_DIR)
    file(READ ${MCCL_INCLUDE_DIR}/mccl.h MCCL_VERSION_FILE_CONTENTS)

    string(REGEX MATCH "define MCCL_MAJOR +([0-9]+)" MCCL_MAJOR_VERSION
                 "${MCCL_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define MCCL_MAJOR +([0-9]+)" "\\1" MCCL_MAJOR_VERSION
                         "${MCCL_MAJOR_VERSION}")
    string(REGEX MATCH "define MCCL_MINOR +([0-9]+)" MCCL_MINOR_VERSION
                 "${MCCL_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define MCCL_MINOR +([0-9]+)" "\\1" MCCL_MINOR_VERSION
                         "${MCCL_MINOR_VERSION}")
    string(REGEX MATCH "define MCCL_PATCH +([0-9]+)" MCCL_PATCH_VERSION
                 "${MCCL_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define MCCL_PATCH +([0-9]+)" "\\1" MCCL_PATCH_VERSION
                         "${MCCL_PATCH_VERSION}")
    if(NOT MCCL_MAJOR_VERSION)
      set(MCCL_VERSION "???")
    else()
      math(EXPR MCCL_VERSION "${MCCL_MAJOR_VERSION} * 1000 +
                 ${MCCL_MINOR_VERSION} * 100 + ${MCCL_PATCH_VERSION}")
    endif()
    include_directories(${MCCL_INCLUDE_DIR})

    message(STATUS "Current MCCL header is ${MCCL_INCLUDE_DIR}/mccl.h. ")
    message(
      STATUS
        "Current MCCL version is "
        "v${MCCL_MAJOR_VERSION}.${MCCL_MINOR_VERSION}.${MCCL_PATCH_VERSION} ")
  else()
    message(FATAL_ERROR "WITH_MCCL  is enabled but mccl.h file is not found!")
  endif()
endif()
