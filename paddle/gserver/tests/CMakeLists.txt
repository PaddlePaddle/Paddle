# gserver pacakge unittests
add_simple_unittest(test_LinearChainCRF)
add_simple_unittest(test_RecurrentLayer)

if(NOT MOBILE_INFERENCE)
  add_simple_unittest(test_MultinomialSampler)
endif()

function(gserver_test TARGET)
  add_unittest_without_exec(${TARGET}
      ${TARGET}.cpp
      LayerGradUtil.cpp)
  add_test(NAME ${TARGET}
      COMMAND ${TARGET})
endfunction()

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/concat_dotmul_a.conf
    COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/* ${CMAKE_CURRENT_BINARY_DIR}
)
add_custom_target(copy_gserver_conf ALL DEPENDS concat_dotmul_a.conf)

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
gserver_test(test_MaxPoolingWithMaskOutput)
gserver_test(test_Upsample)

set(PYTHON_PATH 
   ${PADDLE_SOURCE_DIR}/paddle/.set_python_path.sh -d 
   ${PADDLE_BINARY_DIR}/python/:${PADDLE_BINARY_DIR}/paddle/gserver/tests)
function(gserver_test_with_python TARGET)
  add_unittest_without_exec(${TARGET} ${TARGET}.cpp)
  add_test(NAME ${TARGET}
    COMMAND ${PYTHON_PATH} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}
      WORKING_DIRECTORY ${PADDLE_BINARY_DIR}/paddle/)
endfunction()

gserver_test_with_python(test_PyDataProvider2)
if(WITH_PYTHON)
    gserver_test_with_python(test_PyDataProvider)
endif()
if(NOT MOBILE_INFERENCE)
    gserver_test_with_python(test_CompareTwoNets)
    # TODO(yuyang18): There is some bug in test_RecurrentGradientMachine, I will fix it.
    gserver_test_with_python(test_RecurrentGradientMachine)
endif()

########## test_MKLDNN layers and activations ##########
if(WITH_MKLDNN)
    add_unittest_without_exec(test_MKLDNN
        test_MKLDNN.cpp
        MKLDNNTester.cpp
        LayerGradUtil.cpp)
    add_test(NAME test_MKLDNN
        COMMAND ${PYTHON_PATH} ${CMAKE_CURRENT_BINARY_DIR}/test_MKLDNN
            WORKING_DIRECTORY ${PADDLE_BINARY_DIR}/paddle)
endif()

############### test_WarpCTCLayer #######################
if(NOT WITH_DOUBLE AND NOT MOBILE_INFERENCE)
    add_unittest_without_exec(test_WarpCTCLayer
        test_WarpCTCLayer.cpp)
    add_test(NAME test_WarpCTCLayer
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_WarpCTCLayer --warpctc_dir=${WARPCTC_LIB_DIR}
        WORKING_DIRECTORY ${PADDLE_BINARY_DIR}/paddle)
endif()

if(NOT MOBILE_INFERENCE)
    ################## test_Evaluator #############
    add_unittest(test_Evaluator
        test_Evaluator.cpp)
      
    ########### test_NetworkCompare ###############
    add_unittest_without_exec(test_NetworkCompare
        test_NetworkCompare.cpp)
    if(WITH_GPU)
        set(use_gpu true)
    else()
        set(use_gpu false)
    endif()
    add_test(NAME test_NetworkCompare
        COMMAND ${PYTHON_PATH} ${CMAKE_CURRENT_BINARY_DIR}/test_NetworkCompare --use_gpu=${use_gpu}
        WORKING_DIRECTORY ${PADDLE_BINARY_DIR}/paddle)

    ############ test_CompareSparse ################
    add_unittest_without_exec(test_CompareSparse
        test_CompareSparse.cpp)
    if(NOT ON_TRAVIS)
      add_test(NAME test_CompareSparse
        COMMAND ${PYTHON_PATH} ${PADDLE_SOURCE_DIR}/paddle/.set_port.sh -p port -n 6
                ${CMAKE_CURRENT_BINARY_DIR}/test_CompareSparse
        WORKING_DIRECTORY ${PADDLE_BINARY_DIR}/paddle/)
    endif()
endif()
