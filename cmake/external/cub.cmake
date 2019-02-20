if(NOT WITH_GPU)
  return()
endif()

include(ExternalProject)

set(CUB_SOURCE_DIR ${THIRD_PARTY_PATH}/cub)
set(CUB_INCLUDE_DIR ${CUB_SOURCE_DIR}/src/extern_cub)

include_directories(${CUB_INCLUDE_DIR})

ExternalProject_Add(
  extern_cub
  ${EXTERNAL_PROJECT_LOG_ARGS}
  GIT_REPOSITORY "https://github.com/NVlabs/cub.git"
  GIT_TAG        "v1.8.0"
  PREFIX         ${CUB_SOURCE_DIR}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)

if(${CMAKE_VERSION} VERSION_LESS "3.3.0")
  set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/cub_dummy.c)
  file(WRITE ${dummyfile} "const char *dummy = \"${dummyfile}\";")
  add_library(cub STATIC ${dummyfile})
else()
  add_library(cub INTERFACE)
endif()

add_dependencies(cub extern_cub)
