# New custom OP can support Windows/Linux/Mac now
if(WITH_GPU OR APPLE)
    py_test(test_custom_relu_op_setup SRCS test_custom_relu_op_setup.py)
    py_test(test_custom_relu_op_jit SRCS test_custom_relu_op_jit.py)
    py_test(test_custom_relu_model SRCS test_custom_relu_model.py)

    # Compiling shared library will cost some time, but running process is very fast.
    set_tests_properties(test_custom_relu_op_setup PROPERTIES TIMEOUT 250)
    set_tests_properties(test_custom_relu_op_jit PROPERTIES TIMEOUT 180)
    set_tests_properties(test_custom_relu_model PROPERTIES TIMEOUT 180)
endif()

py_test(test_custom_raw_op_kernel_op SRCS test_custom_raw_op_kernel_op.py)
set_tests_properties(test_custom_raw_op_kernel_op PROPERTIES TIMEOUT 180)

# CPU custom op tests: only compile .cc file
py_test(test_dispatch_jit SRCS test_dispatch_jit.py)
py_test(test_multi_out_jit SRCS test_multi_out_jit.py)
py_test(test_custom_attrs_jit SRCS test_custom_attrs_jit.py)
py_test(test_custom_concat SRCS test_custom_concat.py)
py_test(test_custom_conj SRCS test_custom_conj.py)
py_test(test_custom_linear SRCS test_custom_linear.py)
py_test(test_custom_simple_slice SRCS test_custom_simple_slice.py)

# other tests
py_test(test_sysconfig SRCS test_sysconfig.py)
py_test(test_check_abi SRCS test_check_abi.py)

if(APPLE)
    set_tests_properties(test_custom_simple_slice PROPERTIES TIMEOUT 300)
endif()
