if (NOT WITH_XPU)
    return()
endif()

INCLUDE(ExternalProject)
SET(XPU_PROJECT                 "extern_xpu")
SET(XPU_URL    "https://kunlun1.su.bcebos.com/xpu.tar.gz" CACHE STRING "" FORCE)
SET(XPU_SOURCE_DIR              "${THIRD_PARTY_PATH}/xpu")
SET(XPU_DOWNLOAD_DIR            "${XPU_SOURCE_DIR}/src/${XPU_PROJECT}")
SET(XPU_INSTALL_DIR             "${THIRD_PARTY_PATH}/install/xpu")
SET(XPU_API_INC_DIR             "${THIRD_PARTY_PATH}/install/xpu/api/include")
SET(XPU_RUNTIME_INC_DIR         "${THIRD_PARTY_PATH}/install/xpu/runtime/include")
SET(XPU_LIB_DIR                 "${THIRD_PARTY_PATH}/install/xpu/lib")

SET(XPU_API_LIB_NAME            "libxpuapi.so")
SET(XPU_RT_LIB_NAME             "libxpurt.so")
SET(XPU_SIM_LIB_NAME            "libxpusim.so")
SET(XPU_API_LIB                 "${XPU_LIB_DIR}/${XPU_API_LIB_NAME}")
SET(XPU_RT_LIB                  "${XPU_LIB_DIR}/${XPU_RT_LIB_NAME}")
SET(XPU_SIM_LIB                 "${XPU_LIB_DIR}/${XPU_SIM_LIB_NAME}")

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${XPU_INSTALL_DIR}/lib")

INCLUDE_DIRECTORIES(${XPU_API_INC_DIR})
INCLUDE_DIRECTORIES(${XPU_RUNTIME_INC_DIR})

FILE(WRITE ${XPU_DOWNLOAD_DIR}/CMakeLists.txt
  "PROJECT(XPU)\n"
  "cmake_minimum_required(VERSION 3.0)\n"
  "install(DIRECTORY xpu/api xpu/runtime xpu/lib \n"
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

ADD_LIBRARY(shared_xpuapi SHARED IMPORTED GLOBAL)
set_property(TARGET shared_xpuapi PROPERTY IMPORTED_LOCATION "${XPU_API_LIB}")

# generate a static dummy target to track xpulib dependencies
# for cc_library(xxx SRCS xxx.c DEPS xpulib)
generate_dummy_static_lib(LIB_NAME "xpulib" GENERATOR "xpu.cmake")

TARGET_LINK_LIBRARIES(xpulib ${XPU_API_LIB} ${XPU_RT_LIB} ${XPU_SIM_LIB})
ADD_DEPENDENCIES(xpulib ${XPU_PROJECT})
