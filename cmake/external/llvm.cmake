include(FetchContent)

set(LLVM_DOWNLOAD_URL https://paddle-inference-dist.bj.bcebos.com/infrt/llvm_b5149f4e66a49a98b67e8e2de4e24a4af8e2781b.tar.gz)
set(LLVM_MD5 022819bb5760817013cf4b8a37e97d5e)

set(FETCHCONTENT_BASE_DIR ${THIRD_PARTY_PATH}/llvm)
set(FETCHCONTENT_QUIET OFF)
FetchContent_Declare(external_llvm
  URL ${LLVM_DOWNLOAD_URL}
  URL_MD5 ${LLVM_MD5}
  PREFIX ${THIRD_PARTY_PATH}/llvm
  SOURCE_DIR ${THIRD_PARTY_PATH}/install/llvm
)
if (NOT LLVM_PATH)
  FetchContent_GetProperties(external_llvm)
  if (NOT external_llvm_POPULATED)
    FetchContent_Populate(external_llvm)
  endif()
  set(LLVM_PATH ${THIRD_PARTY_PATH}/install/llvm)
  set(LLVM_DIR ${THIRD_PARTY_PATH}/install/llvm/lib/cmake/llvm)
  set(MLIR_DIR ${THIRD_PARTY_PATH}/install/llvm/lib/cmake/mlir)
else ()
  set(LLVM_DIR ${LLVM_PATH}/lib/cmake/llvm)
  set(MLIR_DIR ${LLVM_PATH}/lib/cmake/mlir)
endif()

if (${CMAKE_CXX_COMPILER} STREQUAL "clang++")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")
endif()

message(STATUS "set LLVM_DIR: ${LLVM_DIR}")
message(STATUS "set MLIR_DIR: ${MLIR_DIR}")
find_package(LLVM REQUIRED CONFIG HINTS ${LLVM_DIR})
find_package(MLIR REQUIRED CONFIG HINTS ${MLIR_DIR})
find_package(ZLIB REQUIRED)

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

# To build with MLIR, the LLVM is build from source code using the following flags:

#[==[
cmake ../llvm  -G "Unix Makefiles" \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_ZLIB=OFF \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DCMAKE_INSTALL_PREFIX=./install
#]==]
# The matched llvm-project version is b5149f4e66a49a98b67e8e2de4e24a4af8e2781b (currently a temporary commit)

add_definitions(${LLVM_DEFINITIONS})

llvm_map_components_to_libnames(llvm_libs Support Core irreader
        X86 executionengine orcjit mcjit all codegen)

message(STATUS "LLVM libs: ${llvm_libs}")

get_property(mlir_libs GLOBAL PROPERTY MLIR_ALL_LIBS)
message(STATUS "MLIR libs: ${mlir_libs}")
add_definitions(${LLVM_DEFINITIONS})


# The minimum needed libraries for MLIR IR parse and transform.
set(MLIR_IR_LIBS MLIRAnalysis MLIRPass MLIRParser MLIRDialect MLIRIR MLIROptLib)


# tb_base is the name of a xxx.td file (without the .td suffix)
function(mlir_tablegen_on td_base)
  set(options)
  set(oneValueArgs DIALECT)
  cmake_parse_arguments(mlir_tablegen_on "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(LLVM_TARGET_DEFINITIONS ${td_base}.td)
  mlir_tablegen(${td_base}.hpp.inc -gen-op-decls)
  mlir_tablegen(${td_base}.cpp.inc -gen-op-defs)
  if (mlir_tablegen_on_DIALECT)
    mlir_tablegen(${td_base}_dialect.hpp.inc --gen-dialect-decls -dialect=${mlir_tablegen_on_DIALECT})
    mlir_tablegen(${td_base}_dialect.cpp.inc --gen-dialect-defs -dialect=${mlir_tablegen_on_DIALECT})
  endif()
  add_public_tablegen_target(${td_base}_IncGen)
  add_custom_target(${td_base}_inc DEPENDS ${td_base}_IncGen)
endfunction()

function(mlir_add_rewriter td_base)
  set(LLVM_TARGET_DEFINITIONS ${td_base}.td)
  set(LLVM_TARGET_DEPENDS  ${LLVM_TARGET_DEPENDS} ${CMAKE_SOURCE_DIR}/paddle/infrt/dialect/infrt/ir/infrt_base.td)
  mlir_tablegen(${td_base}.cpp.inc -gen-rewriters)
  add_public_tablegen_target(MLIR${td_base}IncGen)
  add_dependencies(mlir-headers MLIR${td_base}IncGen)
endfunction()

# Execute the mlir script with infrt-exec program.
# @name: name of the test
# @script: path to the mlir script file
function (infrt_exec_check name script)
  add_test(NAME ${name}
    COMMAND sh -c "${CMAKE_BINARY_DIR}/paddle/infrt/host_context/infrt-exec -i ${CMAKE_CURRENT_SOURCE_DIR}/${script}| ${LLVM_PATH}/bin/FileCheck  ${CMAKE_CURRENT_SOURCE_DIR}/${script}")
endfunction()
