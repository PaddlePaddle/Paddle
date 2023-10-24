include(ExternalProject)

# gmp-6.2.1 https://gmplib.org/download/gmp/gmp-6.2.1.tar.xz
# cln-1.3.6 https://www.ginac.de/CLN/cln-1.3.6.tar.bz2
# ginac-1.8.1 https://www.ginac.de/ginac-1.8.1.tar.bz2
#  all build with CFLAGS="-fPIC -DPIC" CXXFLAGS="-fPIC -DPIC" --enable-static=yes

set(GINAC_FILE
    "ginac-1.8.1_cln-1.3.6_gmp-6.2.1.tar.gz"
    CACHE STRING "" FORCE)
set(GINAC_DOWNLOAD_URL
    "https://paddle-inference-dist.bj.bcebos.com/CINN/${GINAC_FILE}")
set(GINAC_URL_MD5 ebc3e4b7770dd604777ac3f01bfc8b06)
set(GINAC_DOWNLOAD_DIR ${PADDLE_SOURCE_DIR}/third_party/ginac)
set(GINAC_PREFIX_DIR ${THIRD_PARTY_PATH}/ginac)
set(GINAC_INSTALL_DIR ${THIRD_PARTY_PATH}/install/ginac)

function(download_ginac)
  message(
    STATUS
      "Downloading ${GINAC_DOWNLOAD_URL} to ${GINAC_DOWNLOAD_DIR}/${GINAC_FILE}"
  )
  file(
    DOWNLOAD ${GINAC_DOWNLOAD_URL} ${GINAC_DOWNLOAD_DIR}/${GINAC_FILE}
    EXPECTED_MD5 ${GINAC_URL_MD5}
    STATUS ERR)
  if(ERR EQUAL 0)
    message(STATUS "Download ${GINAC_FILE} success")
  else()
    message(
      FATAL_ERROR
        "Download failed, error: ${ERR}\n You can try downloading ${GINAC_FILE} again"
    )
  endif()
endfunction()

# Download and check ginac.
if(EXISTS ${GINAC_DOWNLOAD_DIR}/${GINAC_FILE})
  file(MD5 ${GINAC_DOWNLOAD_DIR}/${GINAC_FILE} GINAC_MD5)
  if(NOT GINAC_MD5 STREQUAL GINAC_URL_MD5)
    # clean build file
    file(REMOVE_RECURSE ${GINAC_PREFIX_DIR})
    file(REMOVE_RECURSE ${GINAC_INSTALL_DIR})
    download_ginac()
  endif()
else()
  download_ginac()
endif()

ExternalProject_Add(
  external_ginac
  ${EXTERNAL_PROJECT_LOG_ARGS}
  URL ${GINAC_DOWNLOAD_DIR}/${GINAC_FILE}
  URL_MD5 ${GINAC_URL_MD5}
  DOWNLOAD_DIR ${GINAC_DOWNLOAD_DIR}
  PREFIX ${GINAC_PREFIX_DIR}
  SOURCE_DIR ${GINAC_INSTALL_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  UPDATE_COMMAND ""
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS ${GINAC_INSTALL_DIR}/lib/libginac.a
  BUILD_BYPRODUCTS ${GINAC_INSTALL_DIR}/lib/libcln.a
  BUILD_BYPRODUCTS ${GINAC_INSTALL_DIR}/lib/libgmp.a)

add_library(ginac STATIC IMPORTED GLOBAL)
add_dependencies(ginac external_ginac)
set_property(TARGET ginac PROPERTY IMPORTED_LOCATION
                                   ${GINAC_INSTALL_DIR}/lib/libginac.a)
target_link_libraries(ginac INTERFACE ${GINAC_INSTALL_DIR}/lib/libcln.a
                                      ${GINAC_INSTALL_DIR}/lib/libgmp.a)
include_directories(${GINAC_INSTALL_DIR}/include)
