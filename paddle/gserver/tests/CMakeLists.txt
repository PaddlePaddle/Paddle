# gserver pacakge unittests

################### test_ProtoDataProvider ############
add_unittest_without_exec(test_ProtoDataProvider
    test_ProtoDataProvider.cpp)

# test_ProtoDataProvider will mkdir as same name,
# so if WORKING_DIRECTORY is default directory, then
# mkdir will get error.
add_test(NAME test_ProtoDataProvider
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_ProtoDataProvider
    WORKING_DIRECTORY ${PROJ_ROOT}/paddle)

################# test_LayerGrad #######################
add_unittest_without_exec(test_LayerGrad
    test_LayerGrad.cpp
    LayerGradUtil.cpp)
add_test(NAME test_LayerGrad
    COMMAND test_LayerGrad)

################ test_CRFLayerGrad ####################
add_unittest_without_exec(test_CRFLayerGrad
    test_CRFLayerGrad.cpp
    LayerGradUtil.cpp)
add_test(NAME test_CRFLayerGrad
    COMMAND test_CRFLayerGrad)


add_unittest_without_exec(test_ActivationGrad
    test_ActivationGrad.cpp
    LayerGradUtil.cpp)
add_test(NAME test_ActivationGrad
    COMMAND test_ActivationGrad)
################# test_ConvTrans #######################
add_unittest_without_exec(test_ConvTrans
    test_ConvTrans.cpp
    LayerGradUtil.cpp)

add_test(NAME test_ConvTrans
    COMMAND test_ConvTrans)
################# test_PriorBox #######################
add_unittest_without_exec(test_PriorBox
    test_PriorBox.cpp
    LayerGradUtil.cpp)

add_test(NAME test_PriorBox
    COMMAND test_PriorBox)
################# test_ConvUnify #######################
add_unittest_without_exec(test_ConvUnify
    test_ConvUnify.cpp
    LayerGradUtil.cpp)
    
add_test(NAME test_ConvUnify
    COMMAND test_ConvUnify)
################# test_BatchNorm #######################
add_unittest_without_exec(test_BatchNorm
    test_BatchNorm.cpp
    LayerGradUtil.cpp)

add_test(NAME test_BatchNorm
    COMMAND test_BatchNorm)
################## test_Evaluator #######################
add_unittest(test_Evaluator
    test_Evaluator.cpp)

################ test_LinearChainCRF ####################
add_simple_unittest(test_LinearChainCRF)

############## test_MultinomialSampler ###################
add_simple_unittest(test_MultinomialSampler)

############## test_PyDataProvider ########################
if(WITH_PYTHON)
    add_unittest_without_exec(test_PyDataProvider
        test_PyDataProvider.cpp)

    add_test(NAME test_PyDataProvider
        COMMAND .set_python_path.sh -d ./gserver/tests:${PROJ_ROOT}/python/ ${CMAKE_CURRENT_BINARY_DIR}/test_PyDataProvider
        WORKING_DIRECTORY ${PROJ_ROOT}/paddle)
endif()

############### test_RecurrentLayer #######################
add_simple_unittest(test_RecurrentLayer)

############### test_WarpCTCLayer #######################
if(NOT WITH_DOUBLE)
    add_unittest_without_exec(test_WarpCTCLayer
        test_WarpCTCLayer.cpp)

    add_test(NAME test_WarpCTCLayer
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_WarpCTCLayer --warpctc_dir=${WARPCTC_LIB_DIR}
        WORKING_DIRECTORY ${PROJ_ROOT}/paddle)
endif()

############### test_RecurrentGradientMachine ###############
# TODO(yuyang18): There is some bug in test_RecurrentGradientMachine
# I will fix it.
add_unittest_without_exec(test_RecurrentGradientMachine
    test_RecurrentGradientMachine.cpp)
add_test(NAME test_RecurrentGradientMachine
    COMMAND .set_python_path.sh -d
            ${PROJ_ROOT}/python:${PROJ_ROOT}/paddle/gserver/tests
            ${CMAKE_CURRENT_BINARY_DIR}/test_RecurrentGradientMachine
    WORKING_DIRECTORY ${PROJ_ROOT}/paddle)

add_unittest_without_exec(test_NetworkCompare
    test_NetworkCompare.cpp)
if(WITH_GPU)
    add_test(NAME test_NetworkCompare
        COMMAND .set_python_path.sh -d ${PROJ_ROOT}/python ${CMAKE_CURRENT_BINARY_DIR}/test_NetworkCompare --use_gpu=true
        WORKING_DIRECTORY ${PROJ_ROOT}/paddle)
else()
    add_test(NAME test_NetworkCompare
        COMMAND .set_python_path.sh -d ${PROJ_ROOT}/python ${CMAKE_CURRENT_BINARY_DIR}/test_NetworkCompare --use_gpu=false
        WORKING_DIRECTORY ${PROJ_ROOT}/paddle)
endif()


add_unittest_without_exec(test_PyDataProvider2
        test_PyDataProvider2.cpp)

add_test(NAME test_PyDataProvider2
   COMMAND .set_python_path.sh -d ${PROJ_ROOT}/paddle/gserver/tests:${PROJ_ROOT}/python ${CMAKE_CURRENT_BINARY_DIR}/test_PyDataProvider2
        WORKING_DIRECTORY ${PROJ_ROOT}/paddle
)
