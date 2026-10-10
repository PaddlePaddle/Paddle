include(FetchContent)

if(WITH_XPU_CADA)
  # On the KUNLUN M100 (xtrans) backend, xtrans's cudnn runtime drags in its own
  # libLLVM-15.so. If CINN kept its downloaded LLVM-12 (statically linked), the
  # two LLVMs would register the same cl::opt options into one process-global
  # registry and abort. To make cudnn (eager conv/bn/pool) and CINN JIT coexist,
  # CINN reuses the SAME libLLVM-15.so instance cudnn uses. We compile against
  # the matching xtdk LLVM 15.0.7 dev headers, and at link/runtime bind the
  # single shared libLLVM-15.so from XTRANS_ROOT (NOT a second static copy).
  set(XPU_CADA_LLVM15_ROOT
      "/ssd1/zhangxiao/zhangxiao/xtdk-llvm15-ubuntu2004_x86_64"
      CACHE PATH "xtdk LLVM 15.0.7 dev package (headers + cmake config)")
  set(LLVM_PATH ${XPU_CADA_LLVM15_ROOT})
  set(LLVM_DIR ${XPU_CADA_LLVM15_ROOT}/lib/cmake/llvm)
  # xtrans's cudnn-runtime libLLVM-15.so: the one and only instance in-process.
  set(XPU_CADA_LLVM15_SHARED
      "${XTRANS_ROOT}/targets/x86_64-linux/lib/libLLVM-15.so"
      CACHE FILEPATH "shared libLLVM-15.so shared with xtrans cudnn")
else()
  # set(LLVM_DOWNLOAD_URL https://paddle-inference-dist.bj.bcebos.com/CINN/llvm11.tar.gz)
  # set(LLVM_MD5 39d32b6be466781dddf5869318dcba53)

  set(LLVM_DOWNLOAD_URL
      https://paddle-inference-dist.bj.bcebos.com/CINN/llvm11-glibc2.17.tar.gz)
  set(LLVM_MD5 33c7d3cc6d370585381e8d90bd7c2198)

  set(FETCHCONTENT_BASE_DIR ${THIRD_PARTY_PATH}/llvm)
  set(FETCHCONTENT_QUIET OFF)
  FetchContent_Declare(
    external_llvm
    URL ${LLVM_DOWNLOAD_URL}
    URL_MD5 ${LLVM_MD5}
    PREFIX ${THIRD_PARTY_PATH}/llvm SOURCE_DIR ${THIRD_PARTY_PATH}/install/llvm)
  if(NOT LLVM_PATH)
    FetchContent_GetProperties(external_llvm)
    if(NOT external_llvm_POPULATED)
      FetchContent_Populate(external_llvm)
    endif()
    set(LLVM_PATH ${THIRD_PARTY_PATH}/install/llvm)
    set(LLVM_DIR ${THIRD_PARTY_PATH}/install/llvm/lib/cmake/llvm)
    set(MLIR_DIR ${THIRD_PARTY_PATH}/install/llvm/lib/cmake/mlir)
  else()
    set(LLVM_DIR ${LLVM_PATH}/lib/cmake/llvm)
    set(MLIR_DIR ${LLVM_PATH}/lib/cmake/mlir)
  endif()
endif()

if(${CMAKE_CXX_COMPILER} STREQUAL "clang++")
  set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")
endif()

message(STATUS "set LLVM_DIR: ${LLVM_DIR}")
message(STATUS "set MLIR_DIR: ${MLIR_DIR}")
find_package(LLVM REQUIRED CONFIG HINTS ${LLVM_DIR})
if(NOT WITH_XPU_CADA)
  find_package(MLIR REQUIRED CONFIG HINTS ${MLIR_DIR})
endif()
find_package(ZLIB REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

include_directories(${LLVM_INCLUDE_DIRS})
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
if(NOT WITH_XPU_CADA)
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  include(AddLLVM)
  include(TableGen)
  include(AddMLIR)
  message(STATUS "Found MLIR: ${MLIR_DIR}")
endif()

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# To build with MLIR, the LLVM is build from source code using the following flags:

#[==[
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_ZLIB=OFF \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_ENABLE_TERMINFO=OFF \
  -DCMAKE_INSTALL_PREFIX=./install
#]==]

# The matched llvm-project version is f9dc2b7079350d0fed3bb3775f496b90483c9e42 (currently a temporary commit)
# Update: to build llvm in manylinux docker with glibc-2.17, and use it in manylinux and ubuntu docker,
# the patch https://gist.github.com/zhiqiu/6e8d969176dce13d98fd15338a16265e is needed.

add_definitions(${LLVM_DEFINITIONS})

if(WITH_XPU_CADA)
  # Link the single shared libLLVM-15.so from xtrans (same instance cudnn uses),
  # NOT the static component archives, so the process holds exactly one LLVM and
  # its cl::opt registry is not double-registered against cudnn's LLVM-15.
  # Expose it as an IMPORTED target (like llvm_map_components_to_libnames does
  # for the non-XPU path) so cinn_cc_library's target_link_libraries AND
  # add_dependencies both accept it.
  add_library(xpu_cada_llvm15 SHARED IMPORTED GLOBAL)
  set_target_properties(xpu_cada_llvm15 PROPERTIES IMPORTED_LOCATION
                                                   ${XPU_CADA_LLVM15_SHARED})
  set(llvm_libs xpu_cada_llvm15)
  set(mlir_libs "")
  set(MLIR_IR_LIBS "")
  message(STATUS "WITH_XPU_CADA: CINN uses shared LLVM-15: ${XPU_CADA_LLVM15_SHARED}")
else()
  llvm_map_components_to_libnames(
    llvm_libs
    Support
    Core
    irreader
    X86
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
      MLIRStandardOps
      MLIRPass
      MLIRParser
      MLIRDialect
      MLIRIR
      MLIROptLib)
endif()

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
