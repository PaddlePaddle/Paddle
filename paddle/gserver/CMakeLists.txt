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

if(NOT WITH_MKLML)
    file(GLOB_RECURSE MKL_HEADER RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "MKLPacked*.h")
    file(GLOB_RECURSE MKL_SOURCES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "MKLPacked*.cpp")
    list(REMOVE_ITEM GSERVER_HEADER ${MKL_HEADER})
    list(REMOVE_ITEM GSERVER_SOURCES ${MKL_SOURCES})
    message(STATUS "Skip compiling with MKLPackedLayers")
else()
    message(STATUS "Compile with MKLPackedLayers")
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
         dataproviders/PyDataProvider2.cpp
         dataproviders/PyDataProvider.cpp)

    # Remove useless gradientmachines
    list(REMOVE_ITEM GSERVER_SOURCES
         gradientmachines/MultiNetwork.cpp
         gradientmachines/RecurrentGradientMachine.cpp
         gradientmachines/ParallelNeuralNetwork.cpp
         gradientmachines/GradientMachineMode.cpp
         gradientmachines/MultiGradientMachine.cpp)

    # Remove layers that used in training
    list(REMOVE_ITEM GSERVER_SOURCES
    	 layers/RecurrentLayerGroup.cpp
         layers/CostLayer.cpp
         layers/MultiBoxLossLayer.cpp
         layers/WarpCTCLayer.cpp
         layers/CTCLayer.cpp
         layers/LinearChainCTC.cpp
         layers/PrintLayer.cpp)
    list(REMOVE_ITEM GSERVER_SOURCES
         layers/OuterProdLayer.cpp
         layers/SumToOneNormLayer.cpp
         layers/ConvShiftLayer.cpp
         layers/InterpolationLayer.cpp
         layers/AgentLayer.cpp
         layers/DotMulOperator.cpp
         layers/GruStepLayer.cpp
         layers/LstmStepLayer.cpp
         layers/ConvexCombinationLayer.cpp
         layers/Conv3DLayer.cpp
         layers/DeConv3DLayer.cpp
         layers/CropLayer.cpp
         layers/CrossEntropyOverBeam.cpp
         layers/DataNormLayer.cpp
         layers/FeatureMapExpandLayer.cpp
         layers/HierarchicalSigmoidLayer.cpp
         layers/MultinomialSampler.cpp
         layers/NCELayer.cpp
         layers/KmaxSeqScoreLayer.cpp
         layers/MDLstmLayer.cpp
         layers/MultiplexLayer.cpp
         layers/PadLayer.cpp
         layers/Pool3DLayer.cpp
         layers/ResizeLayer.cpp
         layers/RotateLayer.cpp
         layers/RowConvLayer.cpp
         layers/RowL2NormLayer.cpp
         layers/SamplingIdLayer.cpp
         layers/ScaleShiftLayer.cpp
         layers/SelectiveFullyConnectedLayer.cpp
         layers/SpatialPyramidPoolLayer.cpp
         layers/BilinearInterpLayer.cpp
         layers/ClipLayer.cpp)
endif()

if(WITH_GPU)
    cuda_add_library(paddle_gserver ${GSERVER_SOURCES})
else()
    add_library(paddle_gserver STATIC
        ${GSERVER_SOURCES})
endif()

add_dependencies(paddle_gserver paddle_proto ${external_project_dependencies})
if(WITH_TESTING)
    add_subdirectory(tests)
endif()
