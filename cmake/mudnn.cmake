if(NOT WITH_MUSA)
  return()
endif()

if(WIN32)
  return()
endif()

find_path(
  MUDNN_INCLUDE_DIR mudnn.h
  PATHS ${MUDNN_ROOT} ${MUDNN_ROOT}/include $ENV{MUDNN_ROOT}
        $ENV{MUDNN_ROOT}/include ${MUSA_TOOLKIT_INCLUDE}
  NO_DEFAULT_PATH)

get_filename_component(__libpath_hist ${MUSA_MUSART_LIBRARY} PATH)

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
  ${MUDNN_ROOT}/local/cuda-${MUSA_VERSION}/targets/${TARGET_ARCH}-linux/lib/
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
  PATHS ${MUDNN_CHECK_LIBRARY_DIRS} ${MUDNN_INCLUDE_DIR} ${__libpath_hist}
  NO_DEFAULT_PATH
  DOC "Path to muDNN library.")

if(MUDNN_INCLUDE_DIR AND MUDNN_LIBRARY)
  set(MUDNN_FOUND ON)
else()
  set(MUDNN_FOUND OFF)
endif()

# TODO(@caizhi): enable mudnn finding
#macro(find_cudnn_version cudnn_header_file)
#endmacro()

#if(MUDNN_FOUND)
#  find_mudnn_version(${MUDNN_INCLUDE_DIR}/mudnn.h)
#  if(NOT MUDNN_MAJOR_VERSION)
#    find_mudnn_version(${MUDNN_INCLUDE_DIR}/mudnn_version.h)
#  endif()
#endif()

