# common package contains:
#   * the utilities:
#       * Thread Libs
#       * Memory Manage libs
#       * CommandLine Parser
#       * Logging
#       * Timer/Stats
#   * the math libraries:
#       * Matrix/Vector
#   * the parameter optimizers.
#   * the parameter updater functions.
#
# TODO(yuyang18): separate libs.
#
file(GLOB MATH_HEADERS . *.h)
file(GLOB MATH_SOURCES . *.cpp)

if(NOT WITH_MKLDNN)
    set(DNN_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/MKLDNNMatrix.h")
    set(DNN_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/MKLDNNMatrix.cpp")
    list(REMOVE_ITEM MATH_HEADERS "${DNN_HEADER}")
    list(REMOVE_ITEM MATH_SOURCES "${DNN_SOURCE}")
    message(STATUS "Skip compiling with MKLDNNMatrix")
else()
    message(STATUS "Compile with MKLDNNMatrix")
endif()

if(MOBILE_INFERENCE)
    # Remove sparse
    list(REMOVE_ITEM MATH_HEADERS
         ${CMAKE_CURRENT_SOURCE_DIR}/CpuSparseMatrix.h
         ${CMAKE_CURRENT_SOURCE_DIR}/SparseMatrix.h
         ${CMAKE_CURRENT_SOURCE_DIR}/SparseRowMatrix.h)
    list(REMOVE_ITEM MATH_SOURCES
         ${CMAKE_CURRENT_SOURCE_DIR}/CpuSparseMatrix.cpp
         ${CMAKE_CURRENT_SOURCE_DIR}/SparseMatrix.cpp
         ${CMAKE_CURRENT_SOURCE_DIR}/SparseRowMatrix.cpp)
endif()
set(MATH_SOURCES
    "${PADDLE_SOURCE_DIR}/paddle/math/BaseMatrix.cu"
    "${PADDLE_SOURCE_DIR}/paddle/math/TrainingAlgorithmOp.cu"
    ${MATH_SOURCES})
if(NOT WITH_GPU)
    # then compile BaseMatrix.cu as c++ file
    compile_cu_as_cpp("${PADDLE_SOURCE_DIR}/paddle/math/BaseMatrix.cu")
    compile_cu_as_cpp("${PADDLE_SOURCE_DIR}/paddle/math/TrainingAlgorithmOp.cu")
    add_library(paddle_math STATIC
        ${MATH_SOURCES})
else()
    cuda_add_library(paddle_math ${MATH_SOURCES})
endif()


add_dependencies(paddle_math paddle_proto ${external_project_dependencies})  # depends
if(WITH_TESTING)
    add_subdirectory(tests)
endif()
