file(GLOB TEST_IR_PASSES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_IR_PASSES "${TEST_IR_PASSES}")

if(((NOT WITH_GPU) AND (NOT WITH_ROCM)) OR WIN32 OR APPLE)
  LIST(REMOVE_ITEM TEST_IR_PASSES test_ir_fusion_group_pass)
endif()

foreach(target ${TEST_IR_PASSES})
  py_test_modules(${target} MODULES ${target})
endforeach()

add_subdirectory(inference)

set_tests_properties(test_fuse_resnet_unit PROPERTIES TIMEOUT 120)
