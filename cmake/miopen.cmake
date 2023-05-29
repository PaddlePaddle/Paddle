if(NOT WITH_ROCM)
  return()
endif()

# Now we don't support ROCm on windows
if(WIN32)
  return()
endif()

set(MIOPEN_ROOT
    ${ROCM_PATH}/miopen
    CACHE PATH "MIOPEN ROOT")

find_path(
  MIOPEN_INCLUDE_DIR "miopen/miopen.h"
  PATHS ${MIOPEN_ROOT} ${MIOPEN_ROOT}/include ${MIOPEN_ROOT}/local/include
        $ENV{MIOPEN_ROOT} $ENV{MIOPEN_ROOT}/include
        $ENV{MIOPEN_ROOT}/local/include
  NO_DEFAULT_PATH)

find_library(
  MIOPEN_LIBRARY
  NAMES "libMIOpen.so"
  PATHS ${MIOPEN_ROOT}
        ${MIOPEN_ROOT}/lib
        ${MIOPEN_ROOT}/lib64
        ${__libpath_hist}
        $ENV{MIOPEN_ROOT}
        $ENV{MIOPEN_ROOT}/lib
        $ENV{MIOPEN_ROOT}/lib64
  NO_DEFAULT_PATH
  DOC "Path to MIOpen library.")

if(MIOPEN_INCLUDE_DIR AND MIOPEN_LIBRARY)
  set(MIOPEN_FOUND ON)
else()
  set(MIOPEN_FOUND OFF)
endif()

macro(find_miopen_version miopen_header_file)
  file(READ ${miopen_header_file} MIOPEN_VERSION_FILE_CONTENTS)
  get_filename_component(MIOPEN_LIB_PATH ${MIOPEN_LIBRARY} DIRECTORY)

  string(REGEX MATCH "define MIOPEN_VERSION_MAJOR +([0-9]+)"
               MIOPEN_MAJOR_VERSION "${MIOPEN_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define MIOPEN_VERSION_MAJOR +([0-9]+)" "\\1"
                       MIOPEN_MAJOR_VERSION "${MIOPEN_MAJOR_VERSION}")
  string(REGEX MATCH "define MIOPEN_VERSION_MINOR +([0-9]+)"
               MIOPEN_MINOR_VERSION "${MIOPEN_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define MIOPEN_VERSION_MINOR +([0-9]+)" "\\1"
                       MIOPEN_MINOR_VERSION "${MIOPEN_MINOR_VERSION}")
  string(REGEX MATCH "define MIOPEN_VERSION_PATCH +([0-9]+)"
               MIOPEN_PATCH_VERSION "${MIOPEN_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define MIOPEN_VERSION_PATCH +([0-9]+)" "\\1"
                       MIOPEN_PATCH_VERSION "${MIOPEN_PATCH_VERSION}")
  string(REGEX MATCH "define MIOPEN_VERSION_TWEAK +([0-9]+)"
               MIOPEN_TWEAK_VERSION "${MIOPEN_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define MIOPEN_VERSION_TWEAK +([0-9]+)" "\\1"
                       MIOPEN_TWEAK_VERSION "${MIOPEN_TWEAK_VERSION}")

  if(NOT MIOPEN_MAJOR_VERSION)
    set(MIOPEN_VERSION "???")
  else()
    add_definitions("-DMIOPEN_MAJOR_VERSION=\"${MIOPEN_MAJOR_VERSION}\"")
    math(EXPR MIOPEN_VERSION "${MIOPEN_MAJOR_VERSION} * 1000 +
             ${MIOPEN_MINOR_VERSION} * 10 + ${MIOPEN_PATCH_VERSION}")
    message(
      STATUS "Current MIOpen header is ${MIOPEN_INCLUDE_DIR}/miopen/miopen.h "
             "Current MIOpen version is v${MIOPEN_MAJOR_VERSION}.\
        ${MIOPEN_MINOR_VERSION}.${MIOPEN_PATCH_VERSION}. ")
  endif()
endmacro()

if(MIOPEN_FOUND)
  find_miopen_version(${MIOPEN_INCLUDE_DIR}/miopen/version.h)
endif()
