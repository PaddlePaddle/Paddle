message("fluid_framework_shared ${fluid_framework_shared}")
if (WITH_GPU)
    #nv_library(relu_op_shared SHARED relu_op.cc relu_op.cu DEPS fluid_framework_shared)
    nv_library(relu_op_shared SHARED SRCS relu_op.cc relu_op.cu)
else()
    #cc_library(relu_op_shared SHARED relu_op.cc DEPS fluid_framework_shared)
    cc_library(relu_op_shared SHARED SRCS relu_op.cc)
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

file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

foreach(src ${TEST_OPS})
    py_test(${src} SRCS ${src}.py)
endforeach()
