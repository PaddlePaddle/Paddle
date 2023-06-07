include(ExternalProject)

set(JEMALLOC_DOWNLOAD_DIR
    ${PADDLE_SOURCE_DIR}/third_party/jemalloc/${CMAKE_SYSTEM_NAME})
set(JEMALLOC_PROJECT "extern_jemalloc")
set(JEMALLOC_URL
    https://github.com/jemalloc/jemalloc/releases/download/5.1.0/jemalloc-5.1.0.tar.bz2
)
set(JEMALLOC_SOURCE_DIR ${THIRD_PARTY_PATH}/jemalloc/src/extern_jemalloc)
set(JEMALLOC_INSTALL ${THIRD_PARTY_PATH}/install/jemalloc)
set(JEMALLOC_INCLUDE_DIR ${JEMALLOC_INSTALL}/include)
set(JEMALLOC_DOWNLOAD_DIR "${JEMALLOC_SOURCE_DIR}/src/${JEMALLOC_PROJECT}")

set(JEMALLOC_STATIC_LIBRARIES
    ${THIRD_PARTY_PATH}/install/jemalloc/lib/libjemalloc_pic.a)
set(JEMALLOC_LIBRARIES
    ${THIRD_PARTY_PATH}/install/jemalloc/lib/libjemalloc_pic.a)
set(JEMALLOC_CACHE_FILENAME "jemalloc-5.1.0.tar.bz2")
set(JEMALLOC_URL_MD5 1f47a5aff2d323c317dfa4cf23be1ce4)

function(download_jemalloc)
  message(
    STATUS
      "Downloading ${JEMALLOC_URL} to ${JEMALLOC_DOWNLOAD_DIR}/${JEMALLOC_CACHE_FILENAME}"
  )
  # NOTE: If the version is updated, consider emptying the folder; maybe add timeout
  file(
    DOWNLOAD ${JEMALLOC_URL} ${JEMALLOC_DOWNLOAD_DIR}/${JEMALLOC_CACHE_FILENAME}
    EXPECTED_MD5 ${JEMALLOC_URL_MD5}
    STATUS ERR)
  if(ERR EQUAL 0)
    message(STATUS "Download ${JEMALLOC_CACHE_FILENAME} success")
  else()
    message(
      FATAL_ERROR
        "Download failed, error: ${ERR}\n You can try downloading ${JEMALLOC_CACHE_FILENAME} again"
    )
  endif()
endfunction()

find_file(
  LOCAL_JEMALLOC_LIB_ZIP
  NAMES ${JEMALLOC_CACHE_FILENAME}
  PATHS ${JEMALLOC_DOWNLOAD_DIR}
  NO_DEFAULT_PATH)

if(LOCAL_JEMALLOC_LIB_ZIP)
  file(MD5 ${JEMALLOC_DOWNLOAD_DIR}/${JEMALLOC_CACHE_FILENAME} JEMALLOC_MD5)
  if(NOT JEMALLOC_MD5 EQUAL JEMALLOC_URL_MD5)
    download_jemalloc()
  endif()
else()
  download_jemalloc()
endif()

ExternalProject_Add(
  extern_jemalloc
  PREFIX ${JEMALLOC_SOURCE_DIR}
  URL ${JEMALLOC_DOWNLOAD_DIR}/${DGC_CACHE_FILENAME}
  URL_MD5 ${JEMALLOC_URL_MD5}
  INSTALL_DIR ${JEMALLOC_INSTALL}
  DOWNLOAD_DIR "${JEMALLOC_DOWNLOAD_DIR}"
  SOURCE_DIR ${JEMALLOC_INSTALL_DIR}
  BUILD_COMMAND make
  BUILD_IN_SOURCE 1
  INSTALL_COMMAND make install
  CONFIGURE_COMMAND "${JEMALLOC_DOWNLOAD_DIR}/configure"
                    --prefix=${JEMALLOC_INSTALL} --disable-initial-exec-tls)

add_library(jemalloc STATIC IMPORTED GLOBAL)
set_property(TARGET jemalloc PROPERTY IMPORTED_LOCATION
                                      ${JEMALLOC_STATIC_LIBRARIES})

include_directories(${JEMALLOC_INCLUDE_DIR})
add_dependencies(jemalloc extern_jemalloc)
