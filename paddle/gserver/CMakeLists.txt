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

if(NOT WITH_MKLDNN)
    file(GLOB_RECURSE DNN_HEADER RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "MKLDNN*.h")
    file(GLOB_RECURSE DNN_SOURCES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "MKLDNN*.cpp")
    list(REMOVE_ITEM GSERVER_HEADER ${DNN_HEADER})
    list(REMOVE_ITEM GSERVER_SOURCES ${DNN_SOURCES})
    message(STATUS "Skip compiling with MKLDNNLayers and MKLDNNActivations")
else()
    message(STATUS "Compile with MKLDNNLayers and MKLDNNActivations")
endif()

if(NOT WITH_GPU)
    list(REMOVE_ITEM GSERVER_HEADER
        layers/CudnnConvBaseLayer.h
        layers/CudnnConvLayer.h
        layers/CudnnConvTransLayer.h
        layers/CudnnPoolLayer.h
        layers/CudnnBatchNormLayer.h)

    list(REMOVE_ITEM GSERVER_SOURCES
        layers/CudnnConvBaseLayer.cpp
        layers/CudnnConvLayer.cpp
        layers/CudnnConvTransLayer.cpp
        layers/CudnnPoolLayer.cpp
        layers/CudnnBatchNormLayer.cpp)
    compile_cu_as_cpp(layers/LstmCompute.cu)
    compile_cu_as_cpp(layers/GruCompute.cu)
endif()

if(NOT WITH_PYTHON)
    list(REMOVE_ITEM GSERVER_SOURCES
            dataproviders/PyDataProvider.cpp)
    
    list(REMOVE_ITEM GSERVER_HEADER
            dataproviders/PyDataProvider.h)
endif()

if(MOBILE_INFERENCE)
    # Remove evaluators
    list(REMOVE_ITEM GSERVER_SOURCES
         layers/ValidationLayer.cpp
         evaluators/Evaluator.cpp
         evaluators/DetectionMAPEvaluator.cpp
         evaluators/CTCErrorEvaluator.cpp
         evaluators/ChunkEvaluator.cpp)

    # Remove dataproviders
    list(REMOVE_ITEM GSERVER_SOURCES
         dataproviders/DataProvider.cpp
         dataproviders/MultiDataProvider.cpp
         dataproviders/ProtoDataProvider.cpp
         dataproviders/PyDataProvider2.cpp
         dataproviders/PyDataProvider.cpp)

    # Remove useless gradientmachines
    list(REMOVE_ITEM GSERVER_SOURCES
         gradientmachines/MultiNetwork.cpp
         gradientmachines/RecurrentGradientMachine.cpp
         gradientmachines/ParallelNeuralNetwork.cpp
         gradientmachines/GradientMachineMode.cpp
         gradientmachines/MultiGradientMachine.cpp)

    # Remove useless layers
    list(REMOVE_ITEM GSERVER_SOURCES
    	 layers/RecurrentLayerGroup.cpp)
endif()

if(WITH_GPU)
    cuda_add_library(paddle_gserver ${GSERVER_SOURCES})
else()
    add_library(paddle_gserver STATIC
        ${GSERVER_SOURCES})
endif()

add_style_check_target(paddle_gserver ${GSERVER_SOURCES})
add_style_check_target(paddle_gserver ${GSERVER_HEADER})
add_dependencies(paddle_gserver paddle_proto ${external_project_dependencies})
if(WITH_TESTING)
    add_subdirectory(tests)
endif()
