include(${PROJECT_SOURCE_DIR}/cmake/architecture.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/cinn/llvm_utils.cmake)

if(LLVM_PATH)
  paddle_resolve_llvm_path(LLVM_PATH)
  set(LLVM_DIR ${LLVM_PATH}/lib/cmake/llvm)
  set(MLIR_DIR ${LLVM_PATH}/lib/cmake/mlir)
endif()

if(${CMAKE_CXX_COMPILER} STREQUAL "clang++")
  set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")
endif()

message(STATUS "set LLVM_DIR: ${LLVM_DIR}")
message(STATUS "set MLIR_DIR: ${MLIR_DIR}")
find_package(LLVM REQUIRED CONFIG HINTS ${LLVM_DIR})
find_package(MLIR REQUIRED CONFIG HINTS ${MLIR_DIR})
find_package(ZLIB REQUIRED)
paddle_fix_llvm_support_tinfo_target()

paddle_get_llvm_native_target(PADDLE_LLVM_NATIVE_TARGET)

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
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_ZLIB=OFF \
  -DLLVM_ENABLE_RTTI=ON \
#]==]
# Use the official LLVM 13.0.1 release package. LLVM 13 includes the
# iterator constructor fix needed by C++20 builds:
# https://github.com/llvm/llvm-project/commit/95d0d8e9e9d1

add_definitions(${LLVM_DEFINITIONS})

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
paddle_select_mlir_standard_lib(PADDLE_MLIR_STANDARD_LIB)
set(MLIR_IR_LIBS
    MLIRAnalysis
    ${PADDLE_MLIR_STANDARD_LIB}
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
