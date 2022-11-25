include(ExternalProject)

set(JEMALLOC_PROJECT "extern_jemalloc")
set(JEMALLOC_URL
    https://github.com/jemalloc/jemalloc/releases/download/5.1.0/jemalloc-5.1.0.tar.bz2
)
set(JEMALLOC_BUILD ${THIRD_PARTY_PATH}/jemalloc/src/extern_jemalloc)
set(JEMALLOC_SOURCE_DIR "${THIRD_PARTY_PATH}/jemalloc")
set(JEMALLOC_INSTALL ${THIRD_PARTY_PATH}/install/jemalloc)
set(JEMALLOC_INCLUDE_DIR ${JEMALLOC_INSTALL}/include)
set(JEMALLOC_DOWNLOAD_DIR "${JEMALLOC_SOURCE_DIR}/src/${JEMALLOC_PROJECT}")

set(JEMALLOC_STATIC_LIBRARIES
    ${THIRD_PARTY_PATH}/install/jemalloc/lib/libjemalloc_pic.a)
set(JEMALLOC_LIBRARIES
    ${THIRD_PARTY_PATH}/install/jemalloc/lib/libjemalloc_pic.a)

ExternalProject_Add(
  extern_jemalloc
  PREFIX ${JEMALLOC_SOURCE_DIR}
  URL ${JEMALLOC_URL}
  INSTALL_DIR ${JEMALLOC_INSTALL}
  DOWNLOAD_DIR "${JEMALLOC_DOWNLOAD_DIR}"
  BUILD_COMMAND $(MAKE)
  BUILD_IN_SOURCE 1
  INSTALL_COMMAND $(MAKE) install
  CONFIGURE_COMMAND "${JEMALLOC_DOWNLOAD_DIR}/configure"
                    --prefix=${JEMALLOC_INSTALL} --disable-initial-exec-tls)

add_library(jemalloc STATIC IMPORTED GLOBAL)
set_property(TARGET jemalloc PROPERTY IMPORTED_LOCATION
                                      ${JEMALLOC_STATIC_LIBRARIES})

include_directories(${JEMALLOC_INCLUDE_DIR})
add_dependencies(jemalloc extern_jemalloc)
