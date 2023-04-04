cmake_minimum_required(VERSION 3.16.0)
project(pass_infra_demo CXX C)

message(STATUS "Building with an installed MLIR.")
if(NOT IS_DIRECTORY ${LLVM_PATH})
  message(
    FATAL_ERROR "Please set LLVM_PATH, e.g. -DLLVM_PATH=llvm-project/build")
endif()

include(ExternalProject)
ExternalProject_Add(
  goolgetest
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/googletest
  BINARY_DIR ${CMAKE_SOURCE_DIR}/third_party/googletest/build
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CMAKE_SOURCE_DIR}/third_party/googletest/install)
set(GTEST_PATH ${CMAKE_SOURCE_DIR}/third_party/googletest/install)
include_directories(${GTEST_PATH}/include)
link_directories(${GTEST_PATH}/lib)

include_directories(${CMAKE_SOURCE_DIR}/third_party/pybind11/include)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++17)

set(LLVM_DIR ${LLVM_PATH}/lib/cmake/llvm)
set(MLIR_DIR ${LLVM_PATH}/lib/cmake/mlir)
list(APPEND CMAKE_MODULE_PATH ${MLIR_DIR} ${LLVM_DIR})
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

get_property(mlir_libs GLOBAL PROPERTY MLIR_ALL_LIBS)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_library(
  pass_infra
  Pass/Pass.cc
  Pass/PassRegistry.cc
  Pass/IRPrinting.cc
  IR/PatternMatch.cc
  Rewrite/FrozenRewritePatternSet.cc
  Rewrite/PatternApplicator.cc
  Transforms/GreedyPatternRewriteDriver.cc
  Pass/PassTiming.cc
  Analysis/DataFlow/Framework.cc
  Analysis/DataFlow/DenseAnalysis.cc)

# python and pybind11
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/pybind11)
pybind11_add_module(pass_python python/pybind.cc)
target_link_libraries(pass_python PUBLIC LLVMSupport MLIRSupport MLIRIR
                                         pass_infra)

add_executable(demo test/demo.cc)
target_link_libraries(demo ${mlir_libs} pass_infra)

add_mlir_pdll_library(MLIRPDLLPatternsIncGen test/patterns.pdll
                      test/patterns.h.pdll.inc)
set(LLVM_TARGET_DEFINITIONS test/patterns.td)
mlir_tablegen(test/patterns.h.td.inc -gen-rewriters)
add_public_tablegen_target(MLIRTDPatternsIncGen)

add_executable(mlir_demo test/mlir_demo.cc)
target_link_libraries(mlir_demo ${mlir_libs})
add_dependencies(mlir_demo MLIRPDLLPatternsIncGen MLIRTDPatternsIncGen)

enable_testing()
add_executable(test_analysis_manager test/AnalysisManagerTest.cc)
target_link_libraries(test_analysis_manager pass_infra gtest gtest_main pthread
                      MLIRIR)

add_test(NAME test_analysis_manager COMMAND test_analysis_manager)
