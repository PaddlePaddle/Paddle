file(GLOB common_srcs "*.cc")

if(WIN32)
  set(COMMON_NAME
      common.dll
      CACHE INTERNAL "" FORCE)
elseif(APPLE)
  set(COMMON_NAME
      libcommon.dylib
      CACHE INTERNAL "" FORCE)
else()
  set(COMMON_NAME
      libcommon.so
      CACHE INTERNAL "" FORCE)
endif()

set(COMMON_LIB
    "${CMAKE_CURRENT_BINARY_DIR}/${COMMON_NAME}"
    CACHE FILEPATH "COMMON Library" FORCE)

set(COMMON_BUILD_TYPE
    SHARED
    CACHE INTERNAL "" FORCE)

cc_library(common ${COMMON_BUILD_TYPE} SRCS ${common_srcs})
if(WIN32)
  set_property(TARGET common PROPERTY WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
# only used for libpaddle_inference.a
cc_library(common_static STATIC SRCS ${common_srcs})
