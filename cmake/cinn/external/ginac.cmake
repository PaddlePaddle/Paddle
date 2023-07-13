include(ExternalProject)

# gmp-6.2.1 https://gmplib.org/download/gmp/gmp-6.2.1.tar.xz
# cln-1.3.6 https://www.ginac.de/CLN/cln-1.3.6.tar.bz2
# ginac-1.8.1 https://www.ginac.de/ginac-1.8.1.tar.bz2
#  all build with CFLAGS="-fPIC -DPIC" CXXFLAGS="-fPIC -DPIC" --enable-static=yes

set(GINAC_DOWNLOAD_URL
    https://paddle-inference-dist.bj.bcebos.com/CINN/ginac-1.8.1_cln-1.3.6_gmp-6.2.1.tar.gz
)
set(GINAC_MD5 ebc3e4b7770dd604777ac3f01bfc8b06)

ExternalProject_Add(
  external_ginac
  ${EXTERNAL_PROJECT_LOG_ARGS}
  URL ${GINAC_DOWNLOAD_URL}
  URL_MD5 ${GINAC_MD5}
  PREFIX ${THIRD_PARTY_PATH}/ginac
  SOURCE_DIR ${THIRD_PARTY_PATH}/install/ginac
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  UPDATE_COMMAND ""
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS ${THIRD_PARTY_PATH}/install/ginac/lib/libginac.a
  BUILD_BYPRODUCTS ${THIRD_PARTY_PATH}/install/ginac/lib/libcln.a
  BUILD_BYPRODUCTS ${THIRD_PARTY_PATH}/install/ginac/lib/libgmp.a)

add_library(ginac STATIC IMPORTED GLOBAL)
add_dependencies(ginac external_ginac)
set_property(
  TARGET ginac PROPERTY IMPORTED_LOCATION
                        ${THIRD_PARTY_PATH}/install/ginac/lib/libginac.a)
target_link_libraries(
  ginac INTERFACE ${THIRD_PARTY_PATH}/install/ginac/lib/libcln.a
                  ${THIRD_PARTY_PATH}/install/ginac/lib/libgmp.a)
include_directories(${THIRD_PARTY_PATH}/install/ginac/include)
