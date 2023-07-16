include(ExternalProject)

# isl https://github.com/inducer/ISL
# commit-id 6a1760fe46967cda2a06387793a6b7d4a0876581
#   depends on llvm f9dc2b7079350d0fed3bb3775f496b90483c9e42
#   depends on gmp-6.2.1
# static build
# CPPFLAGS="-fPIC -DPIC" ./configure --with-gmp-prefix=<gmp-install-path> --with-clang-prefix=<llvm-install-path> --enable-shared=no --enable-static=yes

set(ISL_DOWNLOAD_URL
    https://paddle-inference-dist.bj.bcebos.com/CINN/isl-6a1760fe.tar.gz)
set(ISL_MD5 fff10083fb79d394b8a7b7b2089f6183)

ExternalProject_Add(
  external_isl
  ${EXTERNAL_PROJECT_LOG_ARGS}
  URL ${ISL_DOWNLOAD_URL}
  URL_MD5 ${ISL_MD5}
  PREFIX ${THIRD_PARTY_PATH}/isl
  SOURCE_DIR ${THIRD_PARTY_PATH}/install/isl
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  UPDATE_COMMAND ""
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS ${THIRD_PARTY_PATH}/install/isl/lib/libisl.a)

add_library(isl STATIC IMPORTED GLOBAL)
set_property(TARGET isl PROPERTY IMPORTED_LOCATION
                                 ${THIRD_PARTY_PATH}/install/isl/lib/libisl.a)
add_dependencies(isl external_isl)
include_directories(${THIRD_PARTY_PATH}/install/isl/include)
