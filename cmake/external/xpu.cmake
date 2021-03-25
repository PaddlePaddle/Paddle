if (NOT WITH_XPU)
    return()
endif()

INCLUDE(ExternalProject)
SET(XPU_PROJECT                 "extern_xpu")
SET(XPU_API_LIB_NAME            "libxpuapi.so")
SET(XPU_RT_LIB_NAME             "libxpurt.so")

if(NOT XPU_SDK_ROOT)
  if (WITH_AARCH64)
      SET(XPU_URL    "https://baidu-kunlun-public.su.bcebos.com/paddle_depence/aarch64/xpu_2021_01_13.tar.gz" CACHE STRING "" FORCE)
  elseif(WITH_SUNWAY)
      SET(XPU_URL    "https://baidu-kunlun-public.su.bcebos.com/paddle_depence/sunway/xpu_2021_01_13.tar.gz" CACHE STRING "" FORCE)
  else()
      SET(XPU_URL    "https://baidu-kunlun-public.su.bcebos.com/paddle_depence/xpu_2021_02_27.tar.gz" CACHE STRING "" FORCE)
  endif()

  SET(XPU_SOURCE_DIR              "${THIRD_PARTY_PATH}/xpu")
  SET(XPU_DOWNLOAD_DIR            "${XPU_SOURCE_DIR}/src/${XPU_PROJECT}")
  SET(XPU_INSTALL_DIR             "${THIRD_PARTY_PATH}/install/xpu")
  SET(XPU_API_INC_DIR             "${THIRD_PARTY_PATH}/install/xpu/include")
  SET(XPU_LIB_DIR                 "${THIRD_PARTY_PATH}/install/xpu/lib")

  SET(XPU_API_LIB                 "${XPU_LIB_DIR}/${XPU_API_LIB_NAME}")
  SET(XPU_RT_LIB                  "${XPU_LIB_DIR}/${XPU_RT_LIB_NAME}")

  SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${XPU_INSTALL_DIR}/lib")

  FILE(WRITE ${XPU_DOWNLOAD_DIR}/CMakeLists.txt
    "PROJECT(XPU)\n"
    "cmake_minimum_required(VERSION 3.0)\n"
    "install(DIRECTORY xpu/include xpu/lib \n"
    "        DESTINATION ${XPU_INSTALL_DIR})\n")

  ExternalProject_Add(
      ${XPU_PROJECT}
      ${EXTERNAL_PROJECT_LOG_ARGS}
      PREFIX                ${XPU_SOURCE_DIR}
      DOWNLOAD_DIR          ${XPU_DOWNLOAD_DIR}
      DOWNLOAD_COMMAND      wget --no-check-certificate ${XPU_URL} -c -q -O xpu.tar.gz
                            && tar xvf xpu.tar.gz
      DOWNLOAD_NO_PROGRESS  1
      UPDATE_COMMAND        ""
      CMAKE_ARGS            -DCMAKE_INSTALL_PREFIX=${XPU_INSTALL_ROOT}
      CMAKE_CACHE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=${XPU_INSTALL_ROOT}
  )
else()
  SET(XPU_API_INC_DIR   "${XPU_SDK_ROOT}/XTDK/include/")
  SET(XPU_API_LIB "${XPU_SDK_ROOT}/XTDK/shlib/libxpuapi.so")
  SET(XPU_RT_LIB "${XPU_SDK_ROOT}/XTDK/runtime/shlib/libxpurt.so")
  SET(XPU_LIB_DIR "${XPU_SDK_ROOT}/XTDK/shlib/")
endif()

INCLUDE_DIRECTORIES(${XPU_API_INC_DIR})
ADD_LIBRARY(shared_xpuapi SHARED IMPORTED GLOBAL)
set_property(TARGET shared_xpuapi PROPERTY IMPORTED_LOCATION "${XPU_API_LIB}")

# generate a static dummy target to track xpulib dependencies
# for cc_library(xxx SRCS xxx.c DEPS xpulib)
generate_dummy_static_lib(LIB_NAME "xpulib" GENERATOR "xpu.cmake")

TARGET_LINK_LIBRARIES(xpulib ${XPU_API_LIB} ${XPU_RT_LIB})

if (WITH_XPU_BKCL)
  MESSAGE(STATUS "Compile with XPU BKCL!")
  ADD_DEFINITIONS(-DPADDLE_WITH_XPU_BKCL)

  SET(XPU_BKCL_LIB_NAME         "libbkcl.so")
  SET(XPU_BKCL_LIB              "${XPU_LIB_DIR}/${XPU_BKCL_LIB_NAME}")
  SET(XPU_BKCL_INC_DIR          "${THIRD_PARTY_PATH}/install/xpu/include")
  INCLUDE_DIRECTORIES(${XPU_BKCL_INC_DIR})
  TARGET_LINK_LIBRARIES(xpulib ${XPU_API_LIB} ${XPU_RT_LIB} ${XPU_BKCL_LIB})
else(WITH_XPU_BKCL)
  TARGET_LINK_LIBRARIES(xpulib ${XPU_API_LIB} ${XPU_RT_LIB} )
endif(WITH_XPU_BKCL)

if(NOT XPU_SDK_ROOT)
  ADD_DEPENDENCIES(xpulib ${XPU_PROJECT})
else()
  ADD_CUSTOM_TARGET(extern_xpu DEPENDS xpulib)
endif()

# Ensure that xpu/api.h can be included without dependency errors.
file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/.xpu_headers_dummy.cc CONTENT "")
add_library(xpu_headers_dummy STATIC ${CMAKE_CURRENT_BINARY_DIR}/.xpu_headers_dummy.cc)
add_dependencies(xpu_headers_dummy extern_xpu)
link_libraries(xpu_headers_dummy)
