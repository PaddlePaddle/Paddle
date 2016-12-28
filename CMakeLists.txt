cmake_minimum_required(VERSION 2.8)

project(paddle CXX C)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
set(PROJ_ROOT ${CMAKE_SOURCE_DIR})

find_package(Sphinx)
find_package(Doxygen)
find_package(CUDA QUIET)
find_package(Git REQUIRED)
find_package(Threads REQUIRED)

include(simd)

###################### Configurations ############################
option(WITH_DSO "Compile PaddlePaddle with dynamic linked libraries" ON)
option(WITH_GPU "Compile PaddlePaddle with gpu" ${CUDA_FOUND})
option(WITH_DOUBLE "Compile PaddlePaddle with double precision, otherwise use single precision" OFF)
option(WITH_AVX "Compile PaddlePaddle with avx intrinsics" ${AVX_FOUND})
option(WITH_PYTHON "Compile PaddlePaddle with python interpreter" ON)
option(WITH_STYLE_CHECK "Style Check for PaddlePaddle" ON)
option(WITH_RDMA "Compile PaddlePaddle with rdma support" OFF)
option(WITH_TIMER "Compile PaddlePaddle use timer" OFF)
option(WITH_PROFILER "Compile PaddlePaddle use gpu profiler" OFF)
option(WITH_TESTING "Compile and run unittest for PaddlePaddle" ON)
option(WITH_DOC "Compile PaddlePaddle with documentation" OFF)
option(WITH_SWIG_PY "Compile PaddlePaddle with py PaddlePaddle prediction api" ON)
option(ON_TRAVIS "Running test on travis-ci or not." OFF)
option(ON_COVERALLS "Generating code coverage data on coveralls or not." OFF)
option(COVERALLS_UPLOAD "Uploading the generated coveralls json." ON)

include(external/zlib)      # download, build, install zlib
include(external/gflags)    # download, build, install gflags
include(external/glog)      # download, build, install glog
include(external/gtest)     # download, build, install gtest
include(external/protobuf)  # download, build, install protobuf
include(external/openblas)  # download, build, install openblas
include(external/python)    # download, build, install python
include(external/numpy)     # download, build, install numpy
include(external/swig)      # download, build, install swig
include(external/warpctc)   # download, build, install warpctc

include(package)            # set paddle packages
include(cpplint)            # set paddle c++ style
include(ccache)             # set ccache for compilation
include(util)               # set unittest and link libs
include(rdma)               # set rdma libraries
include(flags)              # set paddle compile flags
include(cudnn)              # set cudnn libraries
include(version)            # set PADDLE_VERSION
include(coveralls)          # set code coverage
include(python_module)      # set python module

include(check_packages)     # check configuration
include(definitions)        # add paddle definitions

include_directories("${PROJ_ROOT}")
include_directories("${PROJ_ROOT}/paddle/cuda/include")
include_directories("${CMAKE_CURRENT_BINARY_DIR}/proto")

set(EXTERNAL_LIBS
    # have not include gtest here.
    ${GFLAGS_LIBRARIES}
    ${GLOG_LIBRARIES}
    ${CBLAS_LIBRARIES}
    ${PROTOBUF_LIBRARIES}
    ${ZLIB_LIBRARIES}
)

add_subdirectory(proto)
add_subdirectory(paddle)
add_subdirectory(python)

if(WITH_DOC)
    add_subdirectory(doc)
endif()
