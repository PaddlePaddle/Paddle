include(ExternalProject)
include(${PROJECT_SOURCE_DIR}/cmake/architecture.cmake)

paddle_normalize_target_arch(PADDLE_TARGET_ARCH)

# isl https://github.com/inducer/ISL
# commit-id 6a1760fe46967cda2a06387793a6b7d4a0876581
#   depends on llvm f9dc2b7079350d0fed3bb3775f496b90483c9e42
#   depends on gmp-6.2.1
# static build
# CPPFLAGS="-fPIC -DPIC" ./configure --with-gmp-prefix=<gmp-install-path> --with-clang-prefix=<llvm-install-path> --enable-shared=no --enable-static=yes

if(PADDLE_TARGET_ARCH STREQUAL "aarch64")
  set(ISL_FILE
      "isl-0.26-aarch64.tar.gz"
      CACHE STRING "" FORCE)
  set(ISL_DOWNLOAD_URL
      "https://paddle-inference-dist.cdn.bcebos.com/CINN/${ISL_FILE}"
      CACHE STRING "ARM ISL package URL")
  set(ISL_URL_MD5
      "092950d5944cbe8163413c740f0c611e"
      CACHE STRING "ARM ISL package MD5")
else()
  set(ISL_FILE
      "isl-6a1760fe.tar.gz"
      CACHE STRING "" FORCE)
  set(ISL_DOWNLOAD_URL
      "https://paddle-inference-dist.bj.bcebos.com/CINN/${ISL_FILE}")
  set(ISL_URL_MD5 fff10083fb79d394b8a7b7b2089f6183)
endif()
set(ISL_DOWNLOAD_DIR ${PADDLE_SOURCE_DIR}/third_party/isl)
set(ISL_PREFIX_DIR ${THIRD_PARTY_PATH}/isl)
set(ISL_INSTALL_DIR ${THIRD_PARTY_PATH}/install/isl)

function(download_isl)
  file(MAKE_DIRECTORY "${ISL_DOWNLOAD_DIR}")
  message(
    STATUS "Downloading ${ISL_DOWNLOAD_URL} to ${ISL_DOWNLOAD_DIR}/${ISL_FILE}")
  file(
    DOWNLOAD ${ISL_DOWNLOAD_URL} ${ISL_DOWNLOAD_DIR}/${ISL_FILE}
    EXPECTED_MD5 ${ISL_URL_MD5}
    STATUS ERR)
  if(ERR EQUAL 0)
    message(STATUS "Download ${ISL_FILE} success")
  else()
    message(
      FATAL_ERROR
        "Download failed, error: ${ERR}\n You can try downloading ${ISL_FILE} again"
    )
  endif()
endfunction()

# Download and check isl.
if(EXISTS ${ISL_DOWNLOAD_DIR}/${ISL_FILE})
  file(MD5 ${ISL_DOWNLOAD_DIR}/${ISL_FILE} ISL_HASH)
  if(NOT ISL_HASH STREQUAL ISL_URL_MD5)
    file(REMOVE_RECURSE ${ISL_PREFIX_DIR})
    file(REMOVE_RECURSE ${ISL_INSTALL_DIR})
    download_isl()
  endif()
else()
  download_isl()
endif()

ExternalProject_Add(
  external_isl
  ${EXTERNAL_PROJECT_LOG_ARGS}
  URL ${ISL_DOWNLOAD_DIR}/${ISL_FILE}
  URL_MD5 ${ISL_URL_MD5}
  DOWNLOAD_DIR ${ISL_DOWNLOAD_DIR}
  PREFIX ${ISL_PREFIX_DIR}
  SOURCE_DIR ${ISL_INSTALL_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  UPDATE_COMMAND ""
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS ${ISL_INSTALL_DIR}/lib/libisl.a)

add_library(isl STATIC IMPORTED GLOBAL)
set_property(TARGET isl PROPERTY IMPORTED_LOCATION
                                 ${ISL_INSTALL_DIR}/lib/libisl.a)
add_dependencies(isl external_isl)
include_directories(${ISL_INSTALL_DIR}/include)
