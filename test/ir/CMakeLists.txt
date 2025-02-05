file(
  GLOB TEST_IR_PASSES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_IR_PASSES "${TEST_IR_PASSES}")

list(REMOVE_ITEM TEST_IR_PASSES test_ir_fusion_group_pass)

if(WIN32 AND WIN_UNITTEST_LEVEL LESS 2)
  message(STATUS "Skip tests unrelated to CUDA/TRT")
else()
  foreach(target ${TEST_IR_PASSES})
    if("${target}" STREQUAL test_inference_datatype_pir)
      py_test_modules(${target} MODULES ${target} ENVS FLAGS_enable_pir_api=1)
      set_tests_properties(test_inference_datatype_pir
                           PROPERTIES LABELS "RUN_TYPE=INFER" TIMEOUT 100)
    else()
      py_test_modules(${target} MODULES ${target})
      set_tests_properties(${target} PROPERTIES LABELS "RUN_TYPE=INFER")
    endif()

  endforeach()

  add_subdirectory(pir)
  set_tests_properties(test_fuse_resnet_unit PROPERTIES TIMEOUT 120)
  set_tests_properties(test_convert_to_mixed_precision PROPERTIES TIMEOUT 300)
endif()

add_subdirectory(inference)
