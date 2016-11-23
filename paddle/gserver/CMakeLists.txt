# Gserver package contains:
#   * Layers
#   * Activations
#   * DataProviders
#   * Evaluators
#   * GradientMachines(NeuralNetwork)
file(GLOB_RECURSE GSERVER_HEADER RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.h")
file(GLOB_RECURSE GSERVER_SOURCES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cpp")
set(GSERVER_SOURCES
    layers/LstmCompute.cu
    layers/GruCompute.cu
    ${GSERVER_SOURCES})

macro(filter_test VAR_NAME)
    set(tmp)
    foreach(p IN LISTS ${VAR_NAME})
        if(NOT ${p} MATCHES ".*tests/.*")
             set(tmp ${p} ${tmp})
        endif()
    endforeach()
    set(${VAR_NAME} ${tmp})
endmacro()

filter_test(GSERVER_HEADER)
filter_test(GSERVER_SOURCES)
if(NOT WITH_GPU)
    list(REMOVE_ITEM GSERVER_HEADER
        layers/CudnnConvLayer.h
        layers/CudnnPoolLayer.h
        layers/CudnnBatchNormLayer.h
        layers/NormProjectionLayer.h
        layers/NormLayer.h)

    list(REMOVE_ITEM GSERVER_SOURCES
        layers/CudnnConvLayer.cpp
        layers/CudnnPoolLayer.cpp
        layers/CudnnBatchNormLayer.cpp
        layers/NormProjectionLayer.cpp
        layers/NormLayer.cpp)
    compile_cu_as_cpp(layers/LstmCompute.cu)
    compile_cu_as_cpp(layers/GruCompute.cu)
endif()

if(NOT WITH_PYTHON)
    list(REMOVE_ITEM GSERVER_SOURCES
            dataproviders/PyDataProvider.cpp)
    
    list(REMOVE_ITEM GSERVER_HEADER
            dataproviders/PyDataProvider.h)
endif()

if(WITH_GPU)
    cuda_add_library(paddle_gserver ${GSERVER_SOURCES})
else()
    add_library(paddle_gserver STATIC
        ${GSERVER_SOURCES})
endif()

add_style_check_target(paddle_gserver ${GSERVER_SOURCES})
add_style_check_target(paddle_gserver ${GSERVER_HEADER})
add_dependencies(paddle_gserver gen_proto_cpp)
if(WITH_TESTING)
    add_subdirectory(tests)
endif()
