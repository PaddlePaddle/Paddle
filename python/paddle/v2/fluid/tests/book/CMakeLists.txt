file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

list(REMOVE_ITEM TEST_OPS test_recognize_digits)
py_test(test_recognize_digits_mlp_cpu
  SRCS test_recognize_digits.py
  ARGS mlp)
py_test(test_recognize_digits_mlp_cuda
  SRCS test_recognize_digits.py
  ARGS mlp --use_cuda)
py_test(test_recognize_digits_conv_cpu
  SRCS test_recognize_digits.py
  ARGS conv)
py_test(test_recognize_digits_conv_cuda
  SRCS test_recognize_digits.py
  ARGS conv --use_cuda)
py_test(test_recognize_digits_mlp_cpu_parallel
  SRCS test_recognize_digits.py
  ARGS mlp --parallel)
py_test(test_recognize_digits_mlp_cuda_parallel
  SRCS test_recognize_digits.py
  ARGS mlp --use_cuda --parallel)
py_test(test_recognize_digits_conv_cpu_parallel
  SRCS test_recognize_digits.py
  ARGS conv --parallel)
py_test(test_recognize_digits_conv_cuda_parallel
  SRCS test_recognize_digits.py
  ARGS conv --use_cuda --parallel)

# default test
foreach(src ${TEST_OPS})
    py_test(${src} SRCS ${src}.py)
endforeach()
