# New custom OP can support Windows/Linux now
if(WITH_GPU OR APPLE) 
    # GPU custom op tests: compile both .cc and .cu file
    py_test(test_custom_relu_op_setup SRCS test_custom_relu_op_setup.py)
    py_test(test_custom_relu_op_jit SRCS test_custom_relu_op_jit.py)
    py_test(test_custom_relu_model SRCS test_custom_relu_model.py)

    # Compiling shared library will cost some time, but running process is very fast.
    set_tests_properties(test_custom_relu_op_setup PROPERTIES TIMEOUT 250)
    set_tests_properties(test_custom_relu_op_jit PROPERTIES TIMEOUT 180)
    set_tests_properties(test_custom_relu_model PROPERTIES TIMEOUT 180)
endif()

# CPU custom op tests: only compile .cc file
py_test(test_dispatch_jit SRCS test_dispatch_jit.py)
py_test(test_multi_out_jit SRCS test_multi_out_jit.py)
py_test(test_custom_attrs_jit SRCS test_custom_attrs_jit.py)
py_test(test_custom_concat SRCS test_custom_concat.py)
py_test(test_custom_conj SRCS test_custom_conj.py)

# other tests
py_test(test_sysconfig SRCS test_sysconfig.py)
py_test(test_check_abi SRCS test_check_abi.py)
cc_test(test_check_error SRCS test_check_error.cc DEPS gtest)
