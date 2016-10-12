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
set(MATH_SOURCES
    "${PROJ_ROOT}/paddle/math/BaseMatrix.cu"
    "${PROJ_ROOT}/paddle/math/TrainingAlgorithmOp.cu"
    ${MATH_SOURCES})
if(NOT WITH_GPU)
    # then compile BaseMatrix.cu as c++ file
    compile_cu_as_cpp("${PROJ_ROOT}/paddle/math/BaseMatrix.cu")
    compile_cu_as_cpp("${PROJ_ROOT}/paddle/math/TrainingAlgorithmOp.cu")
    add_library(paddle_math STATIC
        ${MATH_SOURCES})
else()
    cuda_add_library(paddle_math ${MATH_SOURCES})
endif()



add_style_check_target(paddle_math ${MATH_SOURCES})
add_style_check_target(paddle_math ${MATH_HEADERS})

add_dependencies(paddle_math gen_proto_cpp)  # depends
if(WITH_TESTING)
    add_subdirectory(tests)
endif()
