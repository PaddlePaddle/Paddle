include(ExternalProject)

set(DLPACK_SOURCE_DIR ${THIRD_PARTY_PATH}/dlpack)
set(DLPACK_INCLUDE_DIR ${DLPACK_SOURCE_DIR}/src/extern_dlpack/include)

include_directories(${DLPACK_INCLUDE_DIR})

ExternalProject_Add(
  extern_dlpack
  ${EXTERNAL_PROJECT_LOG_ARGS}
  GIT_REPOSITORY "https://github.com/dmlc/dlpack.git"
  GIT_TAG        "v0.2"
  PREFIX         ${DLPACK_SOURCE_DIR}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)

if(${CMAKE_VERSION} VERSION_LESS "3.3.0")
  set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/dlpack_dummy.c)
  file(WRITE ${dummyfile} "const char *dummy = \"${dummyfile}\";")
  add_library(dlpack STATIC ${dummyfile})
else()
  add_library(dlpack INTERFACE)
endif()

add_dependencies(dlpack extern_dlpack)
