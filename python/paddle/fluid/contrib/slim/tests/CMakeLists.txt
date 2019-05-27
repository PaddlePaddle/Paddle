file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

# NOTE: TODOOOOOOOOOOO
# temporarily disable test_distillation_strategy since it always failed on a specified machine with 4 GPUs
# Need to figure out the root cause and then add it back
list(REMOVE_ITEM TEST_OPS test_distillation_strategy)

if(APPLE OR WIN32 OR NOT WITH_MKL)
    list(REMOVE_ITEM TEST_OPS test_mkldnn_int8_quantization_strategy)
endif()

foreach(src ${TEST_OPS})
    py_test(${src} SRCS ${src}.py)
endforeach()
