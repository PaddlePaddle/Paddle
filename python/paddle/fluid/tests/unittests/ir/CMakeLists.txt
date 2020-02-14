file(GLOB TEST_IR_PASSES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_ir_fc_*.py")
string(REPLACE ".py" "" TEST_IR_PASSES "${TEST_IR_PASSES}")

file(GLOB TEST_IR_ELEMENTWISE_PASSES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_ir_elementwise_*.py")
string(REPLACE ".py" "" TEST_IR_PASSES "${TEST_IR_PASSES}")

if(WITH_GPU AND NOT APPLE AND NOT WIN32)
	foreach(target ${TEST_TRT_IR_ELEMENTWISE_PASSES})
    py_test_modules(${target} MODULES ${target})
  endforeach()
endif()

foreach(target ${TEST_IR_PASSES})
  py_test_modules(${target} MODULES ${target})
endforeach()

add_subdirectory(inference)
