if(NOT WITH_MUSA)
  return()
endif()

if(WIN32)
  return()
else()
  set(MUDNN_ROOT
      "/usr/local/musa"
      CACHE PATH "MUDNN ROOT")
endif()

find_path(
  MUDNN_INCLUDE_DIR mudnn.h
  PATHS ${MUDNN_ROOT} ${MUDNN_ROOT}/include $ENV{MUDNN_ROOT}
        $ENV{MUDNN_ROOT}/include ${MUSA_TOOLKIT_INCLUDE}
  NO_DEFAULT_PATH)

set(TARGET_ARCH "x86_64")
if(NOT ${CMAKE_SYSTEM_PROCESSOR})
  set(TARGET_ARCH ${CMAKE_SYSTEM_PROCESSOR})
endif()

list(
  APPEND
  MUDNN_CHECK_LIBRARY_DIRS
  ${MUDNN_ROOT}
  ${MUDNN_ROOT}/lib64
  ${MUDNN_ROOT}/lib
  ${MUDNN_ROOT}/lib/x64
  ${MUDNN_ROOT}/lib/${TARGET_ARCH}-linux-gnu
  $ENV{MUDNN_ROOT}
  $ENV{MUDNN_ROOT}/lib64
  $ENV{MUDNN_ROOT}/lib
  $ENV{MUDNN_ROOT}/lib/x64
  /usr/lib
  ${MUSA_TOOLKIT_ROOT_DIR}
  ${MUSA_TOOLKIT_ROOT_DIR}/lib/x64)
set(MUDNN_LIB_NAME "")

if(LINUX)
  set(MUDNN_LIB_NAME "libmudnn.so")
endif()

find_library(
  MUDNN_LIBRARY
  NAMES ${MUDNN_LIB_NAME}
  PATHS ${MUDNN_CHECK_LIBRARY_DIRS} ${MUDNN_INCLUDE_DIR}
  NO_DEFAULT_PATH
  DOC "Path to muDNN library.")

if(MUDNN_INCLUDE_DIR AND MUDNN_LIBRARY)
  set(MUDNN_FOUND ON)
else()
  set(MUDNN_FOUND OFF)
endif()

macro(find_mudnn_version mudnn_version_file)
  file(READ ${mudnn_version_file} MUDNN_VERSION_FILE_CONTENTS)
  get_filename_component(MUDNN_LIB_PATH ${MUDNN_LIBRARY} DIRECTORY)

  string(REGEX MATCH "define MUDNN_VERSION_MAJOR +([0-9]+)" MUDNN_MAJOR_VERSION
               "${MUDNN_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define MUDNN_VERSION_MAJOR +([0-9]+)" "\\1"
               MUDNN_MAJOR_VERSION "${MUDNN_MAJOR_VERSION}")
  string(REGEX MATCH "define MUDNN_VERSION_MINOR +([0-9]+)" MUDNN_MINOR_VERSION
               "${MUDNN_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define MUDNN_VERSION_MINOR +([0-9]+)" "\\1"
	       MUDNN_MINOR_VERSION "${MUDNN_MINOR_VERSION}")
  string(REGEX MATCH "define MUDNN_VERSION_PATCH +([0-9]+)" MUDNN_PATCH_VERSION
               "${MUDNN_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define MUDNN_VERSION_PATCH +([0-9]+)" "\\1"
               MUDNN_PATCH_VERSION "${MUDNN_PATCH_VERSION}")

  if(NOT MUDNN_MAJOR_VERSION)
    set(MUDNN_VERSION "???")
  else()
    add_definitions("-DMUDNN_MAJOR_VERSION=\"${MUDNN_MAJOR_VERSION}\"")
    math(EXPR MUDNN_VERSION "${MUDNN_MAJOR_VERSION} * 1000 +
               ${MUDNN_MINOR_VERSION} * 100 + ${MUDNN_PATCH_VERSION}")
    message(STATUS "Current muDNN version file is ${mudnn_version_file} ")
    message(
      STATUS
        "Current muDNN version is v${MUDNN_MAJOR_VERSION}.${MUDNN_MINOR_VERSION}.${MUDNN_PATCH_VERSION}. "
    )
  endif()
endmacro()

if(MUDNN_FOUND)
  find_mudnn_version(${MUDNN_INCLUDE_DIR}/mudnn_version.h)
  include_directories(${MUDNN_INCLUDE_DIR})
endif()
