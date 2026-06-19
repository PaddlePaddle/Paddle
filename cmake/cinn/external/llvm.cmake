include(FetchContent)
include(${PROJECT_SOURCE_DIR}/cmake/architecture.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/cinn/llvm_utils.cmake)

set(FETCHCONTENT_BASE_DIR ${THIRD_PARTY_PATH}/llvm)
set(FETCHCONTENT_QUIET OFF)
paddle_get_llvm_native_target(PADDLE_LLVM_NATIVE_TARGET)
paddle_normalize_target_arch(PADDLE_TARGET_ARCH)

if(PADDLE_TARGET_ARCH STREQUAL "aarch64")
  set(LLVM_DOWNLOAD_URL
      "https://xly-devops.bj.bcebos.com/gouzil/llvm13-aarch64-glibc2.27.tar.gz")
  set(LLVM_SHA256
      6de076472823efa9266d669373b0de620e988a9dd241df94a319da06e7069958)
else()
  set(LLVM_DOWNLOAD_URL
      "https://xly-devops.bj.bcebos.com/gouzil/llvm13-glibc2.27.tar.gz")
  set(LLVM_SHA256
      8e6afb8f51baed5530b1757aff5761a65963a3bd2c76bdb7431967634277086e)
endif()

if(NOT LLVM_PATH)
  FetchContent_Declare(
    external_llvm
    URL ${LLVM_DOWNLOAD_URL}
    URL_HASH SHA256=${LLVM_SHA256}
    PREFIX ${THIRD_PARTY_PATH}/llvm SOURCE_DIR ${THIRD_PARTY_PATH}/install/llvm)
  FetchContent_GetProperties(external_llvm)
  if(NOT external_llvm_POPULATED)
    FetchContent_Populate(external_llvm)
  endif()
  set(LLVM_PATH ${THIRD_PARTY_PATH}/install/llvm)
endif()
paddle_resolve_llvm_path(LLVM_PATH)

set(LLVM_DIR ${LLVM_PATH}/lib/cmake/llvm)
set(MLIR_DIR ${LLVM_PATH}/lib/cmake/mlir)

if(${CMAKE_CXX_COMPILER} STREQUAL "clang++")
  set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")
endif()

message(STATUS "set LLVM_DIR: ${LLVM_DIR}")
message(STATUS "set MLIR_DIR: ${MLIR_DIR}")
find_package(LLVM REQUIRED CONFIG HINTS ${LLVM_DIR})
find_package(MLIR REQUIRED CONFIG HINTS ${MLIR_DIR})
find_package(ZLIB REQUIRED)

set(LLVM_CLANG_EXECUTABLE ${LLVM_PATH}/bin/clang++)
set(LLVM_CONFIG_EXECUTABLE ${LLVM_PATH}/bin/llvm-config)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

include_directories(${LLVM_INCLUDE_DIRS})
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddLLVM)
include(TableGen)
include(AddMLIR)

message(STATUS "Found MLIR: ${MLIR_DIR}")
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# The prebuilt LLVM 13.0.1 packages above were built on Ubuntu 18.04
# (glibc 2.27) from source with the following core flags:

#[==[
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_TARGETS_TO_BUILD="<X86 or AArch64>" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_ZLIB=OFF \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_ENABLE_TERMINFO=OFF \
  -DLLVM_ENABLE_LIBEDIT=OFF \
  -DLLVM_ENABLE_LIBXML2=OFF \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DLLVM_INSTALL_UTILS=ON \
  -DCMAKE_INSTALL_PREFIX=./install
#]==]

# Use the LLVM 13.0.1 glibc 2.27 package built with RTTI enabled. LLVM 13
# includes the iterator constructor fix needed by C++20 builds:
# https://github.com/llvm/llvm-project/commit/95d0d8e9e9d1

add_definitions(${LLVM_DEFINITIONS})

# CINN's LLVM backend is used as a host JIT, so keep target components scoped
# to the native target. Calling InitializeAll* in TargetSelect.h would reference
# every configured target and require the corresponding all-target libraries
# from LLVM-Config.cmake, which is unnecessary for host JIT and expands the link
# surface.
# https://github.com/llvm/llvm-project/blob/llvmorg-13.0.1/llvm/include/llvm/Support/TargetSelect.h
# https://github.com/llvm/llvm-project/blob/llvmorg-13.0.1/llvm/cmake/modules/LLVM-Config.cmake
llvm_map_components_to_libnames(
  llvm_libs
  Support
  Core
  irreader
  ${PADDLE_LLVM_NATIVE_TARGET}
  executionengine
  orcjit
  mcjit
  all
  codegen)

message(STATUS "LLVM libs: ${llvm_libs}")

get_property(mlir_libs GLOBAL PROPERTY MLIR_ALL_LIBS)
add_definitions(${LLVM_DEFINITIONS})

# The minimum needed libraries for MLIR IR parse and transform.
set(MLIR_IR_LIBS
    MLIRAnalysis
    MLIRStandard
    MLIRPass
    MLIRParser
    MLIRDialect
    MLIRIR
    MLIROptLib)

# tb_base is the name of a xxx.td file (without the .td suffix)
function(mlir_tablegen_on td_base)
  set(options)
  set(oneValueArgs DIALECT)
  cmake_parse_arguments(mlir_tablegen_on "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  set(LLVM_TARGET_DEFINITIONS ${td_base}.td)
  mlir_tablegen(${td_base}.hpp.inc -gen-op-decls)
  mlir_tablegen(${td_base}.cpp.inc -gen-op-defs)
  if(mlir_tablegen_on_DIALECT)
    mlir_tablegen(${td_base}_dialect.hpp.inc --gen-dialect-decls
                  -dialect=${mlir_tablegen_on_DIALECT})
  endif()
  add_public_tablegen_target(${td_base}_IncGen)
  add_custom_target(${td_base}_inc DEPENDS ${td_base}_IncGen)
endfunction()

function(mlir_add_rewriter td_base)
  set(LLVM_TARGET_DEFINITIONS ${td_base}.td)
  mlir_tablegen(${td_base}.hpp.inc -gen-rewriters
                "-I${CMAKE_SOURCE_DIR}/infrt/dialect/pass")
  add_public_tablegen_target(${td_base}_IncGen)
  add_custom_target(${td_base}_inc DEPENDS ${td_base}_IncGen)
endfunction()
