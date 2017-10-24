INCLUDE(ExternalProject)

SET(NCCL_SOURCE_DIR ${THIRD_PARTY_PATH}/nccl)

INCLUDE_DIRECTORIES(${NCCL_SOURCE_DIR}/src/extern_nccl/src)


if(WITH_DSO)
  # If we use DSO, we do not build nccl, just download the dependencies
  set(NCCL_BUILD_COMMAND "")
  set(NCCL_INSTALL_COMMAND "")
  set(NCCL_INSTALL_DIR "")
else()
  # otherwise, we build nccl and link it.
  set(NCCL_BUILD_COMMAND "make -j 8")
  set(NCCL_INSTALL_COMMAND  "make install")
  SET(NCCL_INSTALL_DIR ${THIRD_PARTY_PATH}/install/nccl)
endif()

ExternalProject_Add(
        extern_nccl
        ${EXTERNAL_PROJECT_LOG_ARGS}
        GIT_REPOSITORY  "https://github.com/NVIDIA/nccl.git"
        GIT_TAG         "v1.3.4-1"
        PREFIX          "${NCCL_SOURCE_DIR}"
        UPDATE_COMMAND  ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     "${NCCL_BUILD_COMMAND}"
        INSTALL_COMMAND   "${NCCL_INSTALL_COMMAND}"
        INSTALL_DIR       "${NCCL_INSTALL_DIR}"
        TEST_COMMAND      ""
)

if (WITH_DSO)
  if (${CMAKE_VERSION} VERSION_LESS "3.3.0")
    set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/lib_any_dummy.c)
    file(WRITE ${dummyfile} "const char * dummy_any = \"${dummyfile}\";")
    add_library(nccl STATIC ${dummyfile})
  else()
    add_library(nccl INTERFACE)
  endif()
else()
  ADD_LIBRARY(nccl STATIC IMPORTED GLOBAL)
  SET_PROPERTY(TARGET nccl PROPERTY IMPORTED_LOCATION
          ${NCCL_INSTALL_DIR}/lib/libnccl.a)
endif()

add_dependencies(nccl extern_nccl)

LIST(APPEND external_project_dependencies nccl)
