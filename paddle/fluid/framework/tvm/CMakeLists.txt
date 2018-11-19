# FIXME(zjl): Seems that TVM have some bugs in codes, suppress some warning here...
set(TVM_EXTRA_COMPILE_FLAGS "-w -Wno-error=ignored-qualifiers -Wno-error=extra -Wno-error=non-virtual-dtor")

function (set_tvm_compile_flags TARGET)
  string(FIND ${TARGET} test TEST_FOUND_POSITION) 
  if (TEST_FOUND_POSITION LESS 0 OR WITH_TESTING)
    set_target_properties(${TARGET} PROPERTIES COMPILE_FLAGS ${TVM_EXTRA_COMPILE_FLAGS}) 
  endif()
endfunction()

cc_library(tvm_runtime SRCS tvm_runtime.cc DEPS tvm tensor device_context)
set_tvm_compile_flags(tvm_runtime)

cc_library(tvm_compile SRCS tvm_compile.cc DEPS tvm_runtime)
set_tvm_compile_flags(tvm_compile)

if (WITH_GPU)
  nv_test(tvm_runtime_test SRCS tvm_runtime_test.cu DEPS tvm_runtime)
  set_tvm_compile_flags(tvm_runtime_test)

  nv_test(tvm_runtime_batch_matmul_test SRCS tvm_runtime_batch_matmul_test.cu DEPS tvm_runtime blas)
  set_tvm_compile_flags(tvm_runtime_batch_matmul_test)
endif()

cc_test(tvm_compile_test SRCS tvm_compile_test.cc DEPS tvm_compile)
set_tvm_compile_flags(tvm_compile_test)
