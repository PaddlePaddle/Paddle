cc_library(tvm_runtime SRCS tvm_runtime.cc DEPS tvm tensor device_context)

cc_library(tvm_compile SRCS tvm_compile.cc DEPS tvm_runtime)
set_target_properties(tvm_compile PROPERTIES COMPILE_FLAGS "-w -Wno-error=ignored-qualifiers -Wno-error=extra -Wno-error=non-virtual-dtor")

nv_test(tvm_runtime_test SRCS tvm_runtime_test.cu DEPS tvm_runtime)
nv_test(tvm_runtime_batch_matmul_test SRCS tvm_runtime_batch_matmul_test.cu DEPS tvm_runtime blas)
