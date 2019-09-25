if (WITH_GPU)
    nv_library(relu_op_shared SHARED SRCS relu_op.cc relu_op.cu DEPS paddle_framework_shared)
else()
    cc_library(relu_op_shared SHARED SRCS relu_op.cc DEPS paddle_framework_shared)
endif()
set_target_properties(relu_op_shared PROPERTIES OUTPUT_NAME relu2_op)
target_link_libraries(relu_op_shared ${FLUID_FRAMEWORK_SHARED_LIB})

# remove the linked glog and gflags when compling relu_op_shared
# otherwise, there is running error:
# ERROR: something wrong with flag 'logtostderr' in file
# 'third_party/glog/src/extern_glog/src/logging.cc'.
# One possibility: file 'third_party/glog/src/extern_glog/src/logging.cc'
# is being linked both statically and dynamically into this executable.
get_target_property(TARGET_LIBRARIES relu_op_shared LINK_LIBRARIES)
LIST(REMOVE_ITEM TARGET_LIBRARIES glog)
LIST(REMOVE_ITEM TARGET_LIBRARIES gflags)
set_property(TARGET relu_op_shared PROPERTY LINK_LIBRARIES  ${TARGET_LIBRARIES} )

file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

foreach(src ${TEST_OPS})
    py_test(${src} SRCS ${src}.py)
endforeach()

#set(INFER_DEP paddle_inference_api paddle_fluid_api ir_pass_manager analysis_predictor)
#message("INFER_DEP: ${INFER_DEP}")

add_executable(test_custom_op_inference test_custom_op_inference.cc)

target_link_libraries(test_custom_op_inference paddle_fluid_shared gtest gflags)
get_target_property(DEPS test_custom_op_inference LINK_LIBRARIES)
message("DEPS: ${DEPS}")
LIST(REMOVE_ITEM DEPS glog)
message("DEPS: ${DEPS}")
set_property(TARGET test_custom_op_inference PROPERTY LINK_LIBRARIES ${DEPS})

add_test(NAME test_custom_op_inference
         COMMAND test_custom_op_inference
         ARGS --dirname=${PADDLE_BINARY_DIR}/python/paddle/fluid/tests/custom_op/
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(test_custom_op_inference PROPERTIES DEPENDS test_custom_op)


#cc_test(test_custom_op_inference
#  SRCS test_custom_op_inference.cc
#  DEPS paddle_fluid_shared
#  ARGS --dirname=${PADDLE_BINARY_DIR}/python/paddle/fluid/tests/custom_op/)
#  DEPS ${INFER_DEP} 

#get_target_property(DEPS test_custom_op_inference LINK_LIBRARIES)
#LIST(REMOVE_ITEM DEPS glog)
#LIST(REMOVE_ITEM DEPS memory)
#LIST(REMOVE_ITEM DEPS lod_tensor)
#LIST(REMOVE_ITEM DEPS paddle_gtest_main)
#message("DEPS: ${DEPS}")
##LIST(REMOVE_ITEM DEPS gflags)
#set_property(TARGET test_custom_op_inference PROPERTY LINK_LIBRARIES ${DEPS})
