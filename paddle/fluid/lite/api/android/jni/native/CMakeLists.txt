# Generate paddle_lite_jni.so
include_directories(${JNI_INCLUDE_DIRS} ${_classDir} ${_stubDir})
lite_cc_library(paddle_lite_jni MODULE 
    SRCS paddle_init_jni.cc paddle_lite_jni.cc tensor_jni.cc
    DEPS light_api_lite cxx_api_lite
       paddle_api_full paddle_api_lite paddle_api_light
    ARM_DEPS ${arm_kernels})
if (APPLE)
    # MacOS only accepts JNI lib ends with .jnilib or .dylib
    set_target_properties(paddle_lite_jni PROPERTIES SUFFIX ".jnilib")
elseif (WIN32)
    # Windows only accepts JNI lib ends with .dll
    set_target_properties(paddle_lite_jni PROPERTIES SUFFIX ".dll")
endif (APPLE)
# Unlike static library, module library has to link target to be able to work
# as a single .so lib.
target_link_libraries(paddle_lite_jni light_api_lite cxx_api_lite
    paddle_api_full paddle_api_lite paddle_api_light ${arm_kernels})
