# New custom OP can support Windows/Linux now
if(WITH_GPU)
    # 'test_custom_relu_op_setup/jit' compile .cc and .cu file
    py_test(test_custom_relu_op_setup SRCS test_custom_relu_op_setup.py)
    py_test(test_custom_relu_op_jit SRCS test_custom_relu_op_jit.py)
    py_test(test_custom_relu_model SRCS test_custom_relu_model.py)

    # Compiling shared library will cost some time, but running process is very fast.
    set_tests_properties(test_custom_relu_op_setup PROPERTIES TIMEOUT 250)
    set_tests_properties(test_custom_relu_op_jit PROPERTIES TIMEOUT 180)
    set_tests_properties(test_custom_relu_model PROPERTIES TIMEOUT 180)
endif()

py_test(test_sysconfig SRCS test_sysconfig.py)

# 'test_dispatch' compile .cc file
py_test(test_dispatch_jit SRCS test_dispatch_jit.py)
set_tests_properties(test_dispatch_jit PROPERTIES TIMEOUT 120)

py_test(test_multi_out_jit SRCS test_multi_out_jit.py)
set_tests_properties(test_multi_out_jit PROPERTIES TIMEOUT 120)

py_test(test_custom_attrs_jit SRCS test_custom_attrs_jit.py)
set_tests_properties(test_custom_attrs_jit PROPERTIES TIMEOUT 120)

py_test(test_custom_concat SRCS test_custom_concat.py)
set_tests_properties(test_custom_concat PROPERTIES TIMEOUT 120)

py_test(test_check_abi SRCS test_check_abi.py)

cc_test(test_check_error SRCS test_check_error.cc DEPS gtest)

if(NOT LINUX)
    return()
endif()

# Old custom OP only support Linux, only run on Linux
py_test(test_custom_op SRCS test_custom_op.py)
py_test(test_jit_load SRCS test_jit_load.py)
py_test(test_setup_install SRCS test_setup_install.py)
py_test(test_setup_build SRCS test_setup_build.py)

set_tests_properties(test_jit_load PROPERTIES TIMEOUT 180)
set_tests_properties(test_setup_install PROPERTIES TIMEOUT 250)
set_tests_properties(test_setup_build PROPERTIES TIMEOUT 180)


if(WITH_ROCM)
    hip_library(relu_op_shared SHARED SRCS relu_op.cc relu_op.cu DEPS paddle_framework_shared)
elseif(WITH_GPU)
    nv_library(relu_op_shared SHARED SRCS relu_op.cc relu_op.cu DEPS paddle_framework_shared)
else()
    cc_library(relu_op_shared SHARED SRCS relu_op.cc DEPS paddle_framework_shared)
endif()
set_target_properties(relu_op_shared PROPERTIES OUTPUT_NAME relu2_op)
target_link_libraries(relu_op_shared ${FLUID_FRAMEWORK_SHARED_LIB})

# remove the linked glog and gflags when compling relu_op_shared
# otherwise, there is running error:
# ERROR: something wrong with flag 'logtostderr' in file
# 'third_party/glog/src/extern_glog/src/logging.cc'.
# One possibility: file 'third_party/glog/src/extern_glog/src/logging.cc'
# is being linked both statically and dynamically into this executable.
get_target_property(TARGET_LIBRARIES relu_op_shared LINK_LIBRARIES)
LIST(REMOVE_ITEM TARGET_LIBRARIES glog)
LIST(REMOVE_ITEM TARGET_LIBRARIES gflags)
set_property(TARGET relu_op_shared PROPERTY LINK_LIBRARIES  ${TARGET_LIBRARIES} )
