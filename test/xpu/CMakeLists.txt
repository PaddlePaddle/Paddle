file(
  GLOB TEST_XPU_IR_PASSES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_xpu_*.py")
string(REPLACE ".py" "" TEST_XPU_IR_PASSES "${TEST_XPU_IR_PASSES}")

if(WITH_XPU)
  foreach(target ${TEST_XPU_IR_PASSES})
    # TODO add py_test_modules or use other function
    # py_test_modules(${target} MODULES ${target})
    set_tests_properties(${target} PROPERTIES LABELS "RUN_TYPE=INFER")
  endforeach()
endif()
