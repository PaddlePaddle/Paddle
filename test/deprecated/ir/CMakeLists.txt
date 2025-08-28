file(
  GLOB TEST_IR_PASSES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_IR_PASSES "${TEST_IR_PASSES}")

if(((NOT WITH_GPU) AND (NOT WITH_ROCM))
   OR WIN32
   OR APPLE)
  list(REMOVE_ITEM TEST_IR_PASSES test_ir_fusion_group_pass)
endif()

if(WIN32 AND WIN_UNITTEST_LEVEL LESS 2)
  message(STATUS "Skip tests unrelated to CUDA/TRT")
else()
  foreach(target ${TEST_IR_PASSES})
    py_test_modules(${target} MODULES ${target})
    set_tests_properties(${target} PROPERTIES LABELS "RUN_TYPE=INFER")
  endforeach()
  add_subdirectory(pir)
endif()
