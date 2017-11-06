# gserver pacakge unittests

add_simple_unittest(test_LinearChainCRF)
add_simple_unittest(test_MultinomialSampler)
add_simple_unittest(test_RecurrentLayer)

function(gserver_test TARGET)
  add_unittest_without_exec(${TARGET}
      ${TARGET}.cpp
      LayerGradUtil.cpp)
  add_test(NAME ${TARGET}
      COMMAND ${TARGET})
endfunction()

gserver_test(test_LayerGrad)
gserver_test(test_CRFLayerGrad)
gserver_test(test_CrossEntropyOverBeamGrad)
gserver_test(test_SeqSliceLayerGrad)
gserver_test(test_ActivationGrad)
gserver_test(test_ConvTrans)
gserver_test(test_PriorBox)
gserver_test(test_DetectionOutput)
gserver_test(test_ConvUnify)
gserver_test(test_BatchNorm)
gserver_test(test_KmaxSeqScore)
gserver_test(test_Expand)

########## test_Mkldnn layers and activations ##########
if(WITH_MKLDNN)
    add_unittest_without_exec(test_MKLDNN
        test_MKLDNN.cpp
        MKLDNNTester.cpp
        LayerGradUtil.cpp)
    add_test(NAME test_MKLDNN
        COMMAND .set_python_path.sh -d ${PADDLE_SOURCE_DIR}/python
            ${CMAKE_CURRENT_BINARY_DIR}/test_MKLDNN
            WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}/paddle)
endif()

############## test_PyDataProvider ########################
if(WITH_PYTHON)
    add_unittest_without_exec(test_PyDataProvider
        test_PyDataProvider.cpp)

    add_test(NAME test_PyDataProvider
        COMMAND .set_python_path.sh -d ./gserver/tests:${PADDLE_SOURCE_DIR}/python/ ${CMAKE_CURRENT_BINARY_DIR}/test_PyDataProvider
        WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}/paddle)
endif()

############### test_WarpCTCLayer #######################
if(NOT WITH_DOUBLE)
    add_unittest_without_exec(test_WarpCTCLayer
        test_WarpCTCLayer.cpp)

    add_test(NAME test_WarpCTCLayer
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_WarpCTCLayer --warpctc_dir=${WARPCTC_LIB_DIR}
        WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}/paddle)
endif()

if(NOT MOBILE_INFERENCE)
################### test_ProtoDataProvider ############
    add_unittest_without_exec(test_ProtoDataProvider
        test_ProtoDataProvider.cpp)

    # test_ProtoDataProvider will mkdir as same name,
    # so if WORKING_DIRECTORY is default directory, then
    # mkdir will get error.
    add_test(NAME test_ProtoDataProvider
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_ProtoDataProvider
        WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}/paddle)

################## test_Evaluator #######################
    add_unittest(test_Evaluator
        test_Evaluator.cpp)
      
############### test_RecurrentGradientMachine ###############
    # TODO(yuyang18): There is some bug in test_RecurrentGradientMachine
    # I will fix it.
    add_unittest_without_exec(test_RecurrentGradientMachine
        test_RecurrentGradientMachine.cpp)
    add_test(NAME test_RecurrentGradientMachine
        COMMAND .set_python_path.sh -d
                ${PADDLE_SOURCE_DIR}/python:${PADDLE_SOURCE_DIR}/paddle/gserver/tests
                ${CMAKE_CURRENT_BINARY_DIR}/test_RecurrentGradientMachine
        WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}/paddle)
      
############### test_NetworkCompare ###############
    add_unittest_without_exec(test_NetworkCompare
        test_NetworkCompare.cpp)
    if(WITH_GPU)
        add_test(NAME test_NetworkCompare
            COMMAND .set_python_path.sh -d ${PADDLE_SOURCE_DIR}/python ${CMAKE_CURRENT_BINARY_DIR}/test_NetworkCompare --use_gpu=true
            WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}/paddle)
    else()
        add_test(NAME test_NetworkCompare
            COMMAND .set_python_path.sh -d ${PADDLE_SOURCE_DIR}/python ${CMAKE_CURRENT_BINARY_DIR}/test_NetworkCompare --use_gpu=false
            WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}/paddle)
    endif()
endif()


add_unittest_without_exec(test_PyDataProvider2
        test_PyDataProvider2.cpp)

add_test(NAME test_PyDataProvider2
   COMMAND .set_python_path.sh -d ${PADDLE_SOURCE_DIR}/paddle/gserver/tests:${PADDLE_SOURCE_DIR}/python ${CMAKE_CURRENT_BINARY_DIR}/test_PyDataProvider2
        WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}/paddle
)
